// Copyright 2025 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2025 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use anyhow::Context;
use compaction_event_loop::{
    HummockCompactionEventDispatcher, HummockCompactionEventHandler, HummockCompactionEventLoop,
    HummockCompactorDedicatedEventLoop,
};
use fail::fail_point;
use itertools::Itertools;
use parking_lot::Mutex;
use rand::rng as thread_rng;
use rand::seq::SliceRandom;
use risingwave_common::config::meta::default::compaction_config;
use risingwave_common::util::epoch::Epoch;
use risingwave_hummock_sdk::compact_task::{CompactTask, ReportTask};
use risingwave_hummock_sdk::compaction_group::StateTableId;
use risingwave_hummock_sdk::key_range::KeyRange;
use risingwave_hummock_sdk::level::Levels;
use risingwave_hummock_sdk::sstable_info::SstableInfo;
use risingwave_hummock_sdk::table_stats::{
    PbTableStatsMap, add_prost_table_stats_map, purge_prost_table_stats,
};
use risingwave_hummock_sdk::table_watermark::WatermarkSerdeType;
use risingwave_hummock_sdk::version::{GroupDelta, IntraLevelDelta};
use risingwave_hummock_sdk::{
    CompactionGroupId, HummockCompactionTaskId, HummockSstableId, HummockSstableObjectId,
    HummockVersionId, compact_task_to_string, statistics_compact_task,
};
use risingwave_pb::hummock::compact_task::{TaskStatus, TaskType};
use risingwave_pb::hummock::subscribe_compaction_event_response::Event as ResponseEvent;
use risingwave_pb::hummock::{
    CompactTaskAssignment, CompactionConfig, PbCompactStatus, PbCompactTaskAssignment,
    SubscribeCompactionEventRequest, TableOption, TableSchema, compact_task,
};
use thiserror_ext::AsReport;
use tokio::sync::RwLockWriteGuard;
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::error::SendError;
use tokio::sync::oneshot::Sender;
use tokio::task::JoinHandle;
use tonic::Streaming;
use tracing::warn;

use crate::hummock::compaction::selector::level_selector::PickerInfo;
use crate::hummock::compaction::selector::{
    DynamicLevelSelector, DynamicLevelSelectorCore, LocalSelectorStatistic, ManualCompactionOption,
    ManualCompactionSelector, SpaceReclaimCompactionSelector, TombstoneCompactionSelector,
    TtlCompactionSelector, VnodeWatermarkCompactionSelector,
};
use crate::hummock::compaction::{CompactStatus, CompactionDeveloperConfig, CompactionSelector};
use crate::hummock::error::{Error, Result};
use crate::hummock::manager::transaction::{
    HummockVersionStatsTransaction, HummockVersionTransaction,
};
use crate::hummock::manager::versioning::Versioning;
use crate::hummock::metrics_utils::{
    build_compact_task_level_type_metrics_label, trigger_compact_tasks_stat,
    trigger_local_table_stat,
};
use crate::hummock::model::CompactionGroup;
use crate::hummock::sequence::next_compaction_task_id;
use crate::hummock::{HummockManager, commit_multi_var, start_measure_real_process_timer};
use crate::manager::META_NODE_ID;
use crate::model::BTreeMapTransaction;

pub mod compaction_event_loop;
pub mod compaction_group_manager;
pub mod compaction_group_schedule;

static CANCEL_STATUS_SET: LazyLock<HashSet<TaskStatus>> = LazyLock::new(|| {
    [
        TaskStatus::ManualCanceled,
        TaskStatus::SendFailCanceled,
        TaskStatus::AssignFailCanceled,
        TaskStatus::HeartbeatCanceled,
        TaskStatus::InvalidGroupCanceled,
        TaskStatus::NoAvailMemoryResourceCanceled,
        TaskStatus::NoAvailCpuResourceCanceled,
        TaskStatus::HeartbeatProgressCanceled,
    ]
    .into_iter()
    .collect()
});

type CompactionRequestChannelItem = (CompactionGroupId, compact_task::TaskType);

fn init_selectors() -> HashMap<compact_task::TaskType, Box<dyn CompactionSelector>> {
    let mut compaction_selectors: HashMap<compact_task::TaskType, Box<dyn CompactionSelector>> =
        HashMap::default();
    compaction_selectors.insert(
        compact_task::TaskType::Dynamic,
        Box::<DynamicLevelSelector>::default(),
    );
    compaction_selectors.insert(
        compact_task::TaskType::SpaceReclaim,
        Box::<SpaceReclaimCompactionSelector>::default(),
    );
    compaction_selectors.insert(
        compact_task::TaskType::Ttl,
        Box::<TtlCompactionSelector>::default(),
    );
    compaction_selectors.insert(
        compact_task::TaskType::Tombstone,
        Box::<TombstoneCompactionSelector>::default(),
    );
    compaction_selectors.insert(
        compact_task::TaskType::VnodeWatermark,
        Box::<VnodeWatermarkCompactionSelector>::default(),
    );
    compaction_selectors
}

impl HummockVersionTransaction<'_> {
    fn apply_compact_task(&mut self, compact_task: &CompactTask) {
        let mut version_delta = self.new_delta();
        let trivial_move = compact_task.is_trivial_move_task();
        version_delta.trivial_move = trivial_move;

        let group_deltas = &mut version_delta
            .group_deltas
            .entry(compact_task.compaction_group_id)
            .or_default()
            .group_deltas;
        let mut removed_table_ids_map: BTreeMap<u32, HashSet<HummockSstableId>> =
            BTreeMap::default();

        for level in &compact_task.input_ssts {
            let level_idx = level.level_idx;

            removed_table_ids_map
                .entry(level_idx)
                .or_default()
                .extend(level.table_infos.iter().map(|sst| sst.sst_id));
        }

        for (level_idx, removed_table_ids) in removed_table_ids_map {
            let group_delta = GroupDelta::IntraLevel(IntraLevelDelta::new(
                level_idx,
                0, // default
                removed_table_ids,
                vec![], // default
                0,      // default
                compact_task.compaction_group_version_id,
            ));

            group_deltas.push(group_delta);
        }

        let group_delta = GroupDelta::IntraLevel(IntraLevelDelta::new(
            compact_task.target_level,
            compact_task.target_sub_level_id,
            HashSet::new(), // default
            compact_task.sorted_output_ssts.clone(),
            compact_task.split_weight_by_vnode,
            compact_task.compaction_group_version_id,
        ));

        group_deltas.push(group_delta);
        version_delta.pre_apply();
    }
}

#[derive(Default)]
pub struct Compaction {
    /// Compaction task that is already assigned to a compactor
    pub compact_task_assignment: BTreeMap<HummockCompactionTaskId, PbCompactTaskAssignment>,
    /// `CompactStatus` of each compaction group
    pub compaction_statuses: BTreeMap<CompactionGroupId, CompactStatus>,

    pub _deterministic_mode: bool,
}

impl HummockManager {
    pub async fn get_assigned_compact_task_num(&self) -> u64 {
        self.compaction.read().await.compact_task_assignment.len() as u64
    }

    pub async fn list_compaction_status(
        &self,
    ) -> (Vec<PbCompactStatus>, Vec<CompactTaskAssignment>) {
        let compaction = self.compaction.read().await;
        (
            compaction.compaction_statuses.values().map_into().collect(),
            compaction
                .compact_task_assignment
                .values()
                .cloned()
                .collect(),
        )
    }

    pub async fn get_compaction_scores(
        &self,
        compaction_group_id: CompactionGroupId,
    ) -> Vec<PickerInfo> {
        let (status, levels, group) = {
            let compaction = self.compaction.read().await;
            let versioning = self.versioning.read().await;
            let config_manager = self.compaction_group_manager.read().await;
            match (
                compaction.compaction_statuses.get(&compaction_group_id),
                versioning.current_version.levels.get(&compaction_group_id),
                config_manager.try_get_compaction_group_config(compaction_group_id),
            ) {
                (Some(cs), Some(v), Some(cf)) => (cs.to_owned(), v.to_owned(), cf),
                _ => {
                    return vec![];
                }
            }
        };
        let dynamic_level_core = DynamicLevelSelectorCore::new(
            group.compaction_config,
            Arc::new(CompactionDeveloperConfig::default()),
        );
        let ctx = dynamic_level_core.get_priority_levels(&levels, &status.level_handlers);
        ctx.score_levels
    }
}

impl HummockManager {
    pub fn compaction_event_loop(
        hummock_manager: Arc<Self>,
        compactor_streams_change_rx: UnboundedReceiver<(
            u32,
            Streaming<SubscribeCompactionEventRequest>,
        )>,
    ) -> Vec<(JoinHandle<()>, Sender<()>)> {
        let mut join_handle_vec = Vec::default();

        let hummock_compaction_event_handler =
            HummockCompactionEventHandler::new(hummock_manager.clone());

        let dedicated_event_loop = HummockCompactorDedicatedEventLoop::new(
            hummock_manager.clone(),
            hummock_compaction_event_handler.clone(),
        );

        let (dedicated_event_loop_join_handle, event_tx, shutdown_tx) = dedicated_event_loop.run();
        join_handle_vec.push((dedicated_event_loop_join_handle, shutdown_tx));

        let hummock_compaction_event_dispatcher = HummockCompactionEventDispatcher::new(
            hummock_manager.env.opts.clone(),
            hummock_compaction_event_handler,
            Some(event_tx),
        );

        let event_loop = HummockCompactionEventLoop::new(
            hummock_compaction_event_dispatcher,
            hummock_manager.metrics.clone(),
            compactor_streams_change_rx,
        );

        let (event_loop_join_handle, event_loop_shutdown_tx) = event_loop.run();
        join_handle_vec.push((event_loop_join_handle, event_loop_shutdown_tx));

        join_handle_vec
    }

    pub fn add_compactor_stream(
        &self,
        context_id: u32,
        req_stream: Streaming<SubscribeCompactionEventRequest>,
    ) {
        self.compactor_streams_change_tx
            .send((context_id, req_stream))
            .unwrap();
    }

    pub async fn auto_pick_compaction_group_and_type(
        &self,
    ) -> Option<(CompactionGroupId, compact_task::TaskType)> {
        let mut compaction_group_ids = self.compaction_group_ids().await;
        compaction_group_ids.shuffle(&mut thread_rng());

        for cg_id in compaction_group_ids {
            if let Some(pick_type) = self.compaction_state.auto_pick_type(cg_id) {
                return Some((cg_id, pick_type));
            }
        }

        None
    }

    /// This method will return all compaction group id in a random order and task type. If there are any group block by `write_limit`, it will return a single array with `TaskType::Emergency`.
    /// If these groups get different task-type, it will return all group id with `TaskType::Dynamic` if the first group get `TaskType::Dynamic`, otherwise it will return the single group with other task type.
    async fn auto_pick_compaction_groups_and_type(
        &self,
    ) -> (Vec<CompactionGroupId>, compact_task::TaskType) {
        let mut compaction_group_ids = self.compaction_group_ids().await;
        compaction_group_ids.shuffle(&mut thread_rng());

        let mut normal_groups = vec![];
        for cg_id in compaction_group_ids {
            if let Some(pick_type) = self.compaction_state.auto_pick_type(cg_id) {
                if pick_type == TaskType::Dynamic {
                    normal_groups.push(cg_id);
                } else if normal_groups.is_empty() {
                    return (vec![cg_id], pick_type);
                }
            }
        }
        (normal_groups, TaskType::Dynamic)
    }
}

impl HummockManager {
    pub async fn get_compact_tasks_impl(
        &self,
        compaction_groups: Vec<CompactionGroupId>,
        max_select_count: usize,
        selector: &mut Box<dyn CompactionSelector>,
    ) -> Result<(Vec<CompactTask>, Vec<CompactionGroupId>)> {
        let deterministic_mode = self.env.opts.compaction_deterministic_test;

        let mut compaction_guard = self.compaction.write().await;
        let compaction: &mut Compaction = &mut compaction_guard;
        let mut versioning_guard = self.versioning.write().await;
        let versioning: &mut Versioning = &mut versioning_guard;

        let _timer = start_measure_real_process_timer!(self, "get_compact_tasks_impl");

        let start_time = Instant::now();
        let mut compaction_statuses = BTreeMapTransaction::new(&mut compaction.compaction_statuses);

        let mut compact_task_assignment =
            BTreeMapTransaction::new(&mut compaction.compact_task_assignment);

        let mut version = HummockVersionTransaction::new(
            &mut versioning.current_version,
            &mut versioning.hummock_version_deltas,
            self.env.notification_manager(),
            None,
            &self.metrics,
        );
        // Apply stats changes.
        let mut version_stats = HummockVersionStatsTransaction::new(
            &mut versioning.version_stats,
            self.env.notification_manager(),
        );

        if deterministic_mode {
            version.disable_apply_to_txn();
        }
        let all_versioned_table_schemas = if self.env.opts.enable_dropped_column_reclaim {
            self.metadata_manager
                .catalog_controller
                .get_versioned_table_schemas()
                .await
                .map_err(|e| Error::Internal(e.into()))?
        } else {
            HashMap::default()
        };
        let mut unschedule_groups = vec![];
        let mut trivial_tasks = vec![];
        let mut pick_tasks = vec![];
        let developer_config = Arc::new(CompactionDeveloperConfig::new_from_meta_opts(
            &self.env.opts,
        ));
        'outside: for compaction_group_id in compaction_groups {
            if pick_tasks.len() >= max_select_count {
                break;
            }

            if !version
                .latest_version()
                .levels
                .contains_key(&compaction_group_id)
            {
                continue;
            }

            // When the last table of a compaction group is deleted, the compaction group (and its
            // config) is destroyed as well. Then a compaction task for this group may come later and
            // cannot find its config.
            let group_config = {
                let config_manager = self.compaction_group_manager.read().await;

                match config_manager.try_get_compaction_group_config(compaction_group_id) {
                    Some(config) => config,
                    None => continue,
                }
            };

            // StoredIdGenerator already implements ids pre-allocation by ID_PREALLOCATE_INTERVAL.
            let task_id = next_compaction_task_id(&self.env).await?;

            if !compaction_statuses.contains_key(&compaction_group_id) {
                // lazy initialize.
                compaction_statuses.insert(
                    compaction_group_id,
                    CompactStatus::new(
                        compaction_group_id,
                        group_config.compaction_config.max_level,
                    ),
                );
            }
            let mut compact_status = compaction_statuses.get_mut(compaction_group_id).unwrap();

            let can_trivial_move = matches!(selector.task_type(), TaskType::Dynamic)
                || matches!(selector.task_type(), TaskType::Emergency);

            let mut stats = LocalSelectorStatistic::default();
            let member_table_ids: Vec<_> = version
                .latest_version()
                .state_table_info
                .compaction_group_member_table_ids(compaction_group_id)
                .iter()
                .map(|table_id| table_id.table_id)
                .collect();

            let mut table_id_to_option: HashMap<u32, _> = HashMap::default();

            {
                let guard = self.table_id_to_table_option.read();
                for table_id in &member_table_ids {
                    if let Some(opts) = guard.get(table_id) {
                        table_id_to_option.insert(*table_id, *opts);
                    }
                }
            }

            while let Some(compact_task) = compact_status.get_compact_task(
                version
                    .latest_version()
                    .get_compaction_group_levels(compaction_group_id),
                version
                    .latest_version()
                    .state_table_info
                    .compaction_group_member_table_ids(compaction_group_id),
                task_id as HummockCompactionTaskId,
                &group_config,
                &mut stats,
                selector,
                &table_id_to_option,
                developer_config.clone(),
                &version.latest_version().table_watermarks,
                &version.latest_version().state_table_info,
            ) {
                let target_level_id = compact_task.input.target_level as u32;
                let compaction_group_version_id = version
                    .latest_version()
                    .get_compaction_group_levels(compaction_group_id)
                    .compaction_group_version_id;
                let compression_algorithm = match compact_task.compression_algorithm.as_str() {
                    "Lz4" => 1,
                    "Zstd" => 2,
                    _ => 0,
                };
                let vnode_partition_count = compact_task.input.vnode_partition_count;
                let mut compact_task = CompactTask {
                    input_ssts: compact_task.input.input_levels,
                    splits: vec![KeyRange::inf()],
                    sorted_output_ssts: vec![],
                    task_id,
                    target_level: target_level_id,
                    // only gc delete keys in last level because there may be older version in more bottom
                    // level.
                    gc_delete_keys: version
                        .latest_version()
                        .get_compaction_group_levels(compaction_group_id)
                        .is_last_level(target_level_id),
                    base_level: compact_task.base_level as u32,
                    task_status: TaskStatus::Pending,
                    compaction_group_id: group_config.group_id,
                    compaction_group_version_id,
                    existing_table_ids: member_table_ids.clone(),
                    compression_algorithm,
                    target_file_size: compact_task.target_file_size,
                    table_options: table_id_to_option
                        .iter()
                        .map(|(table_id, table_option)| {
                            (*table_id, TableOption::from(table_option))
                        })
                        .collect(),
                    current_epoch_time: Epoch::now().0,
                    compaction_filter_mask: group_config.compaction_config.compaction_filter_mask,
                    target_sub_level_id: compact_task.input.target_sub_level_id,
                    task_type: compact_task.compaction_task_type,
                    split_weight_by_vnode: vnode_partition_count,
                    max_sub_compaction: group_config.compaction_config.max_sub_compaction,
                    ..Default::default()
                };

                let is_trivial_reclaim = compact_task.is_trivial_reclaim();
                let is_trivial_move = compact_task.is_trivial_move_task();
                if is_trivial_reclaim || (is_trivial_move && can_trivial_move) {
                    let log_label = if is_trivial_reclaim {
                        "TrivialReclaim"
                    } else {
                        "TrivialMove"
                    };
                    let label = if is_trivial_reclaim {
                        "trivial-space-reclaim"
                    } else {
                        "trivial-move"
                    };

                    tracing::debug!(
                        "{} for compaction group {}: input: {:?}, cost time: {:?}",
                        log_label,
                        compact_task.compaction_group_id,
                        compact_task.input_ssts,
                        start_time.elapsed()
                    );
                    compact_task.task_status = TaskStatus::Success;
                    compact_status.report_compact_task(&compact_task);
                    if !is_trivial_reclaim {
                        compact_task
                            .sorted_output_ssts
                            .clone_from(&compact_task.input_ssts[0].table_infos);
                    }
                    update_table_stats_for_vnode_watermark_trivial_reclaim(
                        &mut version_stats.table_stats,
                        &compact_task,
                    );
                    self.metrics
                        .compact_frequency
                        .with_label_values(&[
                            label,
                            &compact_task.compaction_group_id.to_string(),
                            selector.task_type().as_str_name(),
                            "SUCCESS",
                        ])
                        .inc();

                    version.apply_compact_task(&compact_task);
                    trivial_tasks.push(compact_task);
                    if trivial_tasks.len() >= self.env.opts.max_trivial_move_task_count_per_loop {
                        break 'outside;
                    }
                } else {
                    self.calculate_vnode_partition(
                        &mut compact_task,
                        group_config.compaction_config.as_ref(),
                    );
                    let (pk_prefix_table_watermarks, non_pk_prefix_table_watermarks) = version
                        .latest_version()
                        .safe_epoch_table_watermarks(&compact_task.existing_table_ids)
                        .into_iter()
                        .partition(|(_table_id, table_watermarke)| {
                            matches!(
                                table_watermarke.watermark_type,
                                WatermarkSerdeType::PkPrefix
                            )
                        });

                    compact_task.pk_prefix_table_watermarks = pk_prefix_table_watermarks;
                    compact_task.non_pk_prefix_table_watermarks = non_pk_prefix_table_watermarks;

                    compact_task.table_schemas = compact_task
                        .existing_table_ids
                        .iter()
                        .filter_map(|table_id| {
                            let id = (*table_id).try_into().unwrap();
                            all_versioned_table_schemas.get(&id).map(|column_ids| {
                                (
                                    *table_id,
                                    TableSchema {
                                        column_ids: column_ids.clone(),
                                    },
                                )
                            })
                        })
                        .collect();

                    compact_task_assignment.insert(
                        compact_task.task_id,
                        CompactTaskAssignment {
                            compact_task: Some(compact_task.clone().into()),
                            context_id: META_NODE_ID, // deprecated
                        },
                    );

                    pick_tasks.push(compact_task);
                    break;
                }

                stats.report_to_metrics(compaction_group_id, self.metrics.as_ref());
                stats = LocalSelectorStatistic::default();
            }
            if pick_tasks
                .last()
                .map(|task| task.compaction_group_id != compaction_group_id)
                .unwrap_or(true)
            {
                unschedule_groups.push(compaction_group_id);
            }
            stats.report_to_metrics(compaction_group_id, self.metrics.as_ref());
        }

        if !trivial_tasks.is_empty() {
            commit_multi_var!(
                self.meta_store_ref(),
                compaction_statuses,
                compact_task_assignment,
                version,
                version_stats
            )?;
            self.metrics
                .compact_task_batch_count
                .with_label_values(&["batch_trivial_move"])
                .observe(trivial_tasks.len() as f64);

            for trivial_task in &trivial_tasks {
                self.metrics
                    .compact_task_trivial_move_sst_count
                    .with_label_values(&[&trivial_task.compaction_group_id.to_string()])
                    .observe(trivial_task.input_ssts[0].table_infos.len() as _);
            }

            drop(versioning_guard);
        } else {
            // We are using a single transaction to ensure that each task has progress when it is
            // created.
            drop(versioning_guard);
            commit_multi_var!(
                self.meta_store_ref(),
                compaction_statuses,
                compact_task_assignment
            )?;
        }
        drop(compaction_guard);
        if !pick_tasks.is_empty() {
            self.metrics
                .compact_task_batch_count
                .with_label_values(&["batch_get_compact_task"])
                .observe(pick_tasks.len() as f64);
        }

        for compact_task in &mut pick_tasks {
            let compaction_group_id = compact_task.compaction_group_id;

            // Initiate heartbeat for the task to track its progress.
            self.compactor_manager
                .initiate_task_heartbeat(compact_task.clone());

            // this task has been finished.
            compact_task.task_status = TaskStatus::Pending;
            let compact_task_statistics = statistics_compact_task(compact_task);

            let level_type_label = build_compact_task_level_type_metrics_label(
                compact_task.input_ssts[0].level_idx as usize,
                compact_task.input_ssts.last().unwrap().level_idx as usize,
            );

            let level_count = compact_task.input_ssts.len();
            if compact_task.input_ssts[0].level_idx == 0 {
                self.metrics
                    .l0_compact_level_count
                    .with_label_values(&[&compaction_group_id.to_string(), &level_type_label])
                    .observe(level_count as _);
            }

            self.metrics
                .compact_task_size
                .with_label_values(&[&compaction_group_id.to_string(), &level_type_label])
                .observe(compact_task_statistics.total_file_size as _);

            self.metrics
                .compact_task_size
                .with_label_values(&[
                    &compaction_group_id.to_string(),
                    &format!("{} uncompressed", level_type_label),
                ])
                .observe(compact_task_statistics.total_uncompressed_file_size as _);

            self.metrics
                .compact_task_file_count
                .with_label_values(&[&compaction_group_id.to_string(), &level_type_label])
                .observe(compact_task_statistics.total_file_count as _);

            tracing::trace!(
                "For compaction group {}: pick up {} {} sub_level in level {} to compact to target {}. cost time: {:?} compact_task_statistics {:?}",
                compaction_group_id,
                level_count,
                compact_task.input_ssts[0].level_type.as_str_name(),
                compact_task.input_ssts[0].level_idx,
                compact_task.target_level,
                start_time.elapsed(),
                compact_task_statistics
            );
        }

        #[cfg(test)]
        {
            self.check_state_consistency().await;
        }
        pick_tasks.extend(trivial_tasks);
        Ok((pick_tasks, unschedule_groups))
    }

    /// Cancels a compaction task no matter it's assigned or unassigned.
    pub async fn cancel_compact_task(&self, task_id: u64, task_status: TaskStatus) -> Result<bool> {
        fail_point!("fp_cancel_compact_task", |_| Err(Error::MetaStore(
            anyhow::anyhow!("failpoint metastore err")
        )));
        let ret = self
            .cancel_compact_task_impl(vec![task_id], task_status)
            .await?;
        Ok(ret[0])
    }

    pub async fn cancel_compact_tasks(
        &self,
        tasks: Vec<u64>,
        task_status: TaskStatus,
    ) -> Result<Vec<bool>> {
        self.cancel_compact_task_impl(tasks, task_status).await
    }

    async fn cancel_compact_task_impl(
        &self,
        task_ids: Vec<u64>,
        task_status: TaskStatus,
    ) -> Result<Vec<bool>> {
        assert!(CANCEL_STATUS_SET.contains(&task_status));
        let tasks = task_ids
            .into_iter()
            .map(|task_id| ReportTask {
                task_id,
                task_status,
                sorted_output_ssts: vec![],
                table_stats_change: HashMap::default(),
                object_timestamps: HashMap::default(),
            })
            .collect_vec();
        let rets = self.report_compact_tasks(tasks).await?;
        #[cfg(test)]
        {
            self.check_state_consistency().await;
        }
        Ok(rets)
    }

    async fn get_compact_tasks(
        &self,
        mut compaction_groups: Vec<CompactionGroupId>,
        max_select_count: usize,
        selector: &mut Box<dyn CompactionSelector>,
    ) -> Result<(Vec<CompactTask>, Vec<CompactionGroupId>)> {
        fail_point!("fp_get_compact_task", |_| Err(Error::MetaStore(
            anyhow::anyhow!("failpoint metastore error")
        )));
        compaction_groups.shuffle(&mut thread_rng());
        let (mut tasks, groups) = self
            .get_compact_tasks_impl(compaction_groups, max_select_count, selector)
            .await?;
        tasks.retain(|task| {
            if task.task_status == TaskStatus::Success {
                debug_assert!(task.is_trivial_reclaim() || task.is_trivial_move_task());
                false
            } else {
                true
            }
        });
        Ok((tasks, groups))
    }

    pub async fn get_compact_task(
        &self,
        compaction_group_id: CompactionGroupId,
        selector: &mut Box<dyn CompactionSelector>,
    ) -> Result<Option<CompactTask>> {
        fail_point!("fp_get_compact_task", |_| Err(Error::MetaStore(
            anyhow::anyhow!("failpoint metastore error")
        )));

        let (normal_tasks, _) = self
            .get_compact_tasks_impl(vec![compaction_group_id], 1, selector)
            .await?;
        for task in normal_tasks {
            if task.task_status != TaskStatus::Success {
                return Ok(Some(task));
            }
            debug_assert!(task.is_trivial_reclaim() || task.is_trivial_move_task());
        }
        Ok(None)
    }

    pub async fn manual_get_compact_task(
        &self,
        compaction_group_id: CompactionGroupId,
        manual_compaction_option: ManualCompactionOption,
    ) -> Result<Option<CompactTask>> {
        let mut selector: Box<dyn CompactionSelector> =
            Box::new(ManualCompactionSelector::new(manual_compaction_option));
        self.get_compact_task(compaction_group_id, &mut selector)
            .await
    }

    pub async fn report_compact_task(
        &self,
        task_id: u64,
        task_status: TaskStatus,
        sorted_output_ssts: Vec<SstableInfo>,
        table_stats_change: Option<PbTableStatsMap>,
        object_timestamps: HashMap<HummockSstableObjectId, u64>,
    ) -> Result<bool> {
        let rets = self
            .report_compact_tasks(vec![ReportTask {
                task_id,
                task_status,
                sorted_output_ssts,
                table_stats_change: table_stats_change.unwrap_or_default(),
                object_timestamps,
            }])
            .await?;
        Ok(rets[0])
    }

    pub async fn report_compact_tasks(&self, report_tasks: Vec<ReportTask>) -> Result<Vec<bool>> {
        let compaction_guard = self.compaction.write().await;
        let versioning_guard = self.versioning.write().await;

        self.report_compact_tasks_impl(report_tasks, compaction_guard, versioning_guard)
            .await
    }

    /// Finishes or cancels a compaction task, according to `task_status`.
    ///
    /// If `context_id` is not None, its validity will be checked when writing meta store.
    /// Its ownership of the task is checked as well.
    ///
    /// Return Ok(false) indicates either the task is not found,
    /// or the task is not owned by `context_id` when `context_id` is not None.
    pub async fn report_compact_tasks_impl(
        &self,
        report_tasks: Vec<ReportTask>,
        mut compaction_guard: RwLockWriteGuard<'_, Compaction>,
        mut versioning_guard: RwLockWriteGuard<'_, Versioning>,
    ) -> Result<Vec<bool>> {
        let deterministic_mode = self.env.opts.compaction_deterministic_test;
        let compaction: &mut Compaction = &mut compaction_guard;
        let start_time = Instant::now();
        let original_keys = compaction.compaction_statuses.keys().cloned().collect_vec();
        let mut compact_statuses = BTreeMapTransaction::new(&mut compaction.compaction_statuses);
        let mut rets = vec![false; report_tasks.len()];
        let mut compact_task_assignment =
            BTreeMapTransaction::new(&mut compaction.compact_task_assignment);
        // The compaction task is finished.
        let versioning: &mut Versioning = &mut versioning_guard;
        let _timer = start_measure_real_process_timer!(self, "report_compact_tasks");

        // purge stale compact_status
        for group_id in original_keys {
            if !versioning.current_version.levels.contains_key(&group_id) {
                compact_statuses.remove(group_id);
            }
        }
        let mut tasks = vec![];

        let mut version = HummockVersionTransaction::new(
            &mut versioning.current_version,
            &mut versioning.hummock_version_deltas,
            self.env.notification_manager(),
            None,
            &self.metrics,
        );

        if deterministic_mode {
            version.disable_apply_to_txn();
        }

        let mut version_stats = HummockVersionStatsTransaction::new(
            &mut versioning.version_stats,
            self.env.notification_manager(),
        );
        let mut success_count = 0;
        for (idx, task) in report_tasks.into_iter().enumerate() {
            rets[idx] = true;
            let mut compact_task = match compact_task_assignment.remove(task.task_id) {
                Some(compact_task) => CompactTask::from(compact_task.compact_task.unwrap()),
                None => {
                    tracing::warn!("{}", format!("compact task {} not found", task.task_id));
                    rets[idx] = false;
                    continue;
                }
            };

            {
                // apply result
                compact_task.task_status = task.task_status;
                compact_task.sorted_output_ssts = task.sorted_output_ssts;
            }

            match compact_statuses.get_mut(compact_task.compaction_group_id) {
                Some(mut compact_status) => {
                    compact_status.report_compact_task(&compact_task);
                }
                None => {
                    // When the group_id is not found in the compaction_statuses, it means the group has been removed.
                    // The task is invalid and should be canceled.
                    // e.g.
                    // 1. The group is removed by the user unregistering the tables
                    // 2. The group is removed by the group scheduling algorithm
                    compact_task.task_status = TaskStatus::InvalidGroupCanceled;
                }
            }

            let is_success = if let TaskStatus::Success = compact_task.task_status {
                match self
                    .report_compaction_sanity_check(&task.object_timestamps)
                    .await
                {
                    Err(e) => {
                        warn!(
                            "failed to commit compaction task {} {}",
                            compact_task.task_id,
                            e.as_report()
                        );
                        compact_task.task_status = TaskStatus::RetentionTimeRejected;
                        false
                    }
                    _ => {
                        let group = version
                            .latest_version()
                            .levels
                            .get(&compact_task.compaction_group_id)
                            .unwrap();
                        let is_expired = compact_task.is_expired(group.compaction_group_version_id);
                        if is_expired {
                            compact_task.task_status = TaskStatus::InputOutdatedCanceled;
                            warn!(
                                "The task may be expired because of group split, task:\n {:?}",
                                compact_task_to_string(&compact_task)
                            );
                        }
                        !is_expired
                    }
                }
            } else {
                false
            };
            if is_success {
                success_count += 1;
                version.apply_compact_task(&compact_task);
                if purge_prost_table_stats(&mut version_stats.table_stats, version.latest_version())
                {
                    self.metrics.version_stats.reset();
                    versioning.local_metrics.clear();
                }
                add_prost_table_stats_map(&mut version_stats.table_stats, &task.table_stats_change);
                trigger_local_table_stat(
                    &self.metrics,
                    &mut versioning.local_metrics,
                    &version_stats,
                    &task.table_stats_change,
                );
            }
            tasks.push(compact_task);
        }
        if success_count > 0 {
            commit_multi_var!(
                self.meta_store_ref(),
                compact_statuses,
                compact_task_assignment,
                version,
                version_stats
            )?;

            self.metrics
                .compact_task_batch_count
                .with_label_values(&["batch_report_task"])
                .observe(success_count as f64);
        } else {
            // The compaction task is cancelled or failed.
            commit_multi_var!(
                self.meta_store_ref(),
                compact_statuses,
                compact_task_assignment
            )?;
        }

        let mut success_groups = vec![];
        for compact_task in &tasks {
            self.compactor_manager
                .remove_task_heartbeat(compact_task.task_id);
            tracing::trace!(
                "Reported compaction task. {}. cost time: {:?}",
                compact_task_to_string(compact_task),
                start_time.elapsed(),
            );

            if !deterministic_mode
                && (matches!(compact_task.task_type, compact_task::TaskType::Dynamic)
                    || matches!(compact_task.task_type, compact_task::TaskType::Emergency))
            {
                // only try send Dynamic compaction
                self.try_send_compaction_request(
                    compact_task.compaction_group_id,
                    compact_task::TaskType::Dynamic,
                );
            }

            if compact_task.task_status == TaskStatus::Success {
                success_groups.push(compact_task.compaction_group_id);
            }
        }

        trigger_compact_tasks_stat(
            &self.metrics,
            &tasks,
            &compaction.compaction_statuses,
            &versioning_guard.current_version,
        );
        drop(versioning_guard);
        if !success_groups.is_empty() {
            self.try_update_write_limits(&success_groups).await;
        }
        Ok(rets)
    }

    /// Triggers compacitons to specified compaction groups.
    /// Don't wait for compaction finish
    pub async fn trigger_compaction_deterministic(
        &self,
        _base_version_id: HummockVersionId,
        compaction_groups: Vec<CompactionGroupId>,
    ) -> Result<()> {
        self.on_current_version(|old_version| {
            tracing::info!(
                "Trigger compaction for version {}, groups {:?}",
                old_version.id,
                compaction_groups
            );
        })
        .await;

        if compaction_groups.is_empty() {
            return Ok(());
        }
        for compaction_group in compaction_groups {
            self.try_send_compaction_request(compaction_group, compact_task::TaskType::Dynamic);
        }
        Ok(())
    }

    pub async fn trigger_manual_compaction(
        &self,
        compaction_group: CompactionGroupId,
        manual_compaction_option: ManualCompactionOption,
    ) -> Result<()> {
        let start_time = Instant::now();

        // 1. Get idle compactor.
        let compactor = match self.compactor_manager.next_compactor() {
            Some(compactor) => compactor,
            None => {
                tracing::warn!("trigger_manual_compaction No compactor is available.");
                return Err(anyhow::anyhow!(
                    "trigger_manual_compaction No compactor is available. compaction_group {}",
                    compaction_group
                )
                .into());
            }
        };

        // 2. Get manual compaction task.
        let compact_task = self
            .manual_get_compact_task(compaction_group, manual_compaction_option)
            .await;
        let compact_task = match compact_task {
            Ok(Some(compact_task)) => compact_task,
            Ok(None) => {
                // No compaction task available.
                return Err(anyhow::anyhow!(
                    "trigger_manual_compaction No compaction_task is available. compaction_group {}",
                    compaction_group
                )
                    .into());
            }
            Err(err) => {
                tracing::warn!(error = %err.as_report(), "Failed to get compaction task");

                return Err(anyhow::anyhow!(err)
                    .context(format!(
                        "Failed to get compaction task for compaction_group {}",
                        compaction_group,
                    ))
                    .into());
            }
        };

        // 3. send task to compactor
        let compact_task_string = compact_task_to_string(&compact_task);
        // TODO: shall we need to cancel on meta ?
        compactor
            .send_event(ResponseEvent::CompactTask(compact_task.into()))
            .with_context(|| {
                format!(
                    "Failed to trigger compaction task for compaction_group {}",
                    compaction_group,
                )
            })?;

        tracing::info!(
            "Trigger manual compaction task. {}. cost time: {:?}",
            &compact_task_string,
            start_time.elapsed(),
        );

        Ok(())
    }

    /// Sends a compaction request.
    pub fn try_send_compaction_request(
        &self,
        compaction_group: CompactionGroupId,
        task_type: compact_task::TaskType,
    ) -> bool {
        match self
            .compaction_state
            .try_sched_compaction(compaction_group, task_type)
        {
            Ok(_) => true,
            Err(e) => {
                tracing::error!(
                    error = %e.as_report(),
                    "failed to send compaction request for compaction group {}",
                    compaction_group,
                );
                false
            }
        }
    }

    pub(crate) fn calculate_vnode_partition(
        &self,
        compact_task: &mut CompactTask,
        compaction_config: &CompactionConfig,
    ) {
        // do not split sst by vnode partition when target_level > base_level
        // The purpose of data alignment is mainly to improve the parallelism of base level compaction and reduce write amplification.
        // However, at high level, the size of the sst file is often larger and only contains the data of a single table_id, so there is no need to cut it.
        if compact_task.target_level > compact_task.base_level {
            return;
        }
        if compaction_config.split_weight_by_vnode > 0 {
            for table_id in &compact_task.existing_table_ids {
                compact_task
                    .table_vnode_partition
                    .insert(*table_id, compact_task.split_weight_by_vnode);
            }
        } else {
            let mut table_size_info: HashMap<u32, u64> = HashMap::default();
            let mut existing_table_ids: HashSet<u32> = HashSet::default();
            for input_ssts in &compact_task.input_ssts {
                for sst in &input_ssts.table_infos {
                    existing_table_ids.extend(sst.table_ids.iter());
                    for table_id in &sst.table_ids {
                        *table_size_info.entry(*table_id).or_default() +=
                            sst.sst_size / (sst.table_ids.len() as u64);
                    }
                }
            }
            compact_task
                .existing_table_ids
                .retain(|table_id| existing_table_ids.contains(table_id));

            let hybrid_vnode_count = self.env.opts.hybrid_partition_node_count;
            let default_partition_count = self.env.opts.partition_vnode_count;
            // We must ensure the partition threshold large enough to avoid too many small files.
            let compact_task_table_size_partition_threshold_low = self
                .env
                .opts
                .compact_task_table_size_partition_threshold_low;
            let compact_task_table_size_partition_threshold_high = self
                .env
                .opts
                .compact_task_table_size_partition_threshold_high;
            // check latest write throughput
            let table_write_throughput_statistic_manager =
                self.table_write_throughput_statistic_manager.read();
            let timestamp = chrono::Utc::now().timestamp();
            for (table_id, compact_table_size) in table_size_info {
                let write_throughput = table_write_throughput_statistic_manager
                    .get_table_throughput_descending(table_id, timestamp)
                    .peekable()
                    .peek()
                    .map(|item| item.throughput)
                    .unwrap_or(0);
                if compact_table_size > compact_task_table_size_partition_threshold_high
                    && default_partition_count > 0
                {
                    compact_task
                        .table_vnode_partition
                        .insert(table_id, default_partition_count);
                } else if (compact_table_size > compact_task_table_size_partition_threshold_low
                    || (write_throughput > self.env.opts.table_high_write_throughput_threshold
                        && compact_table_size > compaction_config.target_file_size_base))
                    && hybrid_vnode_count > 0
                {
                    // partition for large write throughput table. But we also need to make sure that it can not be too small.
                    compact_task
                        .table_vnode_partition
                        .insert(table_id, hybrid_vnode_count);
                } else if compact_table_size > compaction_config.target_file_size_base {
                    // partition for small table
                    compact_task.table_vnode_partition.insert(table_id, 1);
                }
            }
            compact_task
                .table_vnode_partition
                .retain(|table_id, _| compact_task.existing_table_ids.contains(table_id));
        }
    }

    pub fn compactor_manager_ref(&self) -> crate::hummock::CompactorManagerRef {
        self.compactor_manager.clone()
    }
}

#[cfg(any(test, feature = "test"))]
impl HummockManager {
    pub async fn compaction_task_from_assignment_for_test(
        &self,
        task_id: u64,
    ) -> Option<CompactTaskAssignment> {
        let compaction_guard = self.compaction.read().await;
        let assignment_ref = &compaction_guard.compact_task_assignment;
        assignment_ref.get(&task_id).cloned()
    }

    pub async fn report_compact_task_for_test(
        &self,
        task_id: u64,
        compact_task: Option<CompactTask>,
        task_status: TaskStatus,
        sorted_output_ssts: Vec<SstableInfo>,
        table_stats_change: Option<PbTableStatsMap>,
    ) -> Result<()> {
        if let Some(task) = compact_task {
            let mut guard = self.compaction.write().await;
            guard.compact_task_assignment.insert(
                task_id,
                CompactTaskAssignment {
                    compact_task: Some(task.into()),
                    context_id: 0,
                },
            );
        }

        // In the test, the contents of the compact task may have been modified directly, while the contents of compact_task_assignment were not modified.
        // So we pass the modified compact_task directly into the `report_compact_task_impl`
        self.report_compact_tasks(vec![ReportTask {
            task_id,
            task_status,
            sorted_output_ssts,
            table_stats_change: table_stats_change.unwrap_or_default(),
            object_timestamps: HashMap::default(),
        }])
        .await?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct CompactionState {
    scheduled: Mutex<HashSet<(CompactionGroupId, compact_task::TaskType)>>,
}

impl CompactionState {
    pub fn new() -> Self {
        Self {
            scheduled: Default::default(),
        }
    }

    /// Enqueues only if the target is not yet in queue.
    pub fn try_sched_compaction(
        &self,
        compaction_group: CompactionGroupId,
        task_type: TaskType,
    ) -> std::result::Result<bool, SendError<CompactionRequestChannelItem>> {
        let mut guard = self.scheduled.lock();
        let key = (compaction_group, task_type);
        if guard.contains(&key) {
            return Ok(false);
        }
        guard.insert(key);
        Ok(true)
    }

    pub fn unschedule(
        &self,
        compaction_group: CompactionGroupId,
        task_type: compact_task::TaskType,
    ) {
        self.scheduled.lock().remove(&(compaction_group, task_type));
    }

    pub fn auto_pick_type(&self, group: CompactionGroupId) -> Option<TaskType> {
        let guard = self.scheduled.lock();
        if guard.contains(&(group, compact_task::TaskType::Dynamic)) {
            Some(compact_task::TaskType::Dynamic)
        } else if guard.contains(&(group, compact_task::TaskType::SpaceReclaim)) {
            Some(compact_task::TaskType::SpaceReclaim)
        } else if guard.contains(&(group, compact_task::TaskType::Ttl)) {
            Some(compact_task::TaskType::Ttl)
        } else if guard.contains(&(group, compact_task::TaskType::Tombstone)) {
            Some(compact_task::TaskType::Tombstone)
        } else if guard.contains(&(group, compact_task::TaskType::VnodeWatermark)) {
            Some(compact_task::TaskType::VnodeWatermark)
        } else {
            None
        }
    }
}

impl Compaction {
    pub fn get_compact_task_assignments_by_group_id(
        &self,
        compaction_group_id: CompactionGroupId,
    ) -> Vec<CompactTaskAssignment> {
        self.compact_task_assignment
            .iter()
            .filter_map(|(_, assignment)| {
                if assignment
                    .compact_task
                    .as_ref()
                    .is_some_and(|task| task.compaction_group_id == compaction_group_id)
                {
                    Some(CompactTaskAssignment {
                        compact_task: assignment.compact_task.clone(),
                        context_id: assignment.context_id,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Clone, Default)]
pub struct CompactionGroupStatistic {
    pub group_id: CompactionGroupId,
    pub group_size: u64,
    pub table_statistic: BTreeMap<StateTableId, u64>,
    pub compaction_group_config: CompactionGroup,
}

/// Updates table stats caused by vnode watermark trivial reclaim compaction.
fn update_table_stats_for_vnode_watermark_trivial_reclaim(
    table_stats: &mut PbTableStatsMap,
    task: &CompactTask,
) {
    if task.task_type != TaskType::VnodeWatermark {
        return;
    }
    let mut deleted_table_keys: HashMap<u32, u64> = HashMap::default();
    for s in task.input_ssts.iter().flat_map(|l| l.table_infos.iter()) {
        assert_eq!(s.table_ids.len(), 1);
        let e = deleted_table_keys.entry(s.table_ids[0]).or_insert(0);
        *e += s.total_key_count;
    }
    for (table_id, delete_count) in deleted_table_keys {
        let Some(stats) = table_stats.get_mut(&table_id) else {
            continue;
        };
        if stats.total_key_count == 0 {
            continue;
        }
        let new_total_key_count = stats.total_key_count.saturating_sub(delete_count as i64);
        let ratio = new_total_key_count as f64 / stats.total_key_count as f64;
        // total_key_count is updated accurately.
        stats.total_key_count = new_total_key_count;
        // others are updated approximately.
        stats.total_key_size = (stats.total_key_size as f64 * ratio).ceil() as i64;
        stats.total_value_size = (stats.total_value_size as f64 * ratio).ceil() as i64;
    }
}

#[derive(Debug, Clone)]
pub enum GroupState {
    /// The compaction group is not in emergency state.
    Normal,

    /// The compaction group is in emergency state.
    Emergency(String), // reason

    /// The compaction group is in write stop state.
    WriteStop(String), // reason
}

impl GroupState {
    pub fn is_write_stop(&self) -> bool {
        matches!(self, Self::WriteStop(_))
    }

    pub fn is_emergency(&self) -> bool {
        matches!(self, Self::Emergency(_))
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Emergency(reason) | Self::WriteStop(reason) => Some(reason),
            _ => None,
        }
    }
}

#[derive(Clone, Default)]
pub struct GroupStateValidator;

impl GroupStateValidator {
    pub fn write_stop_sub_level_count(
        level_count: usize,
        compaction_config: &CompactionConfig,
    ) -> bool {
        let threshold = compaction_config.level0_stop_write_threshold_sub_level_number as usize;
        level_count > threshold
    }

    pub fn write_stop_l0_size(l0_size: u64, compaction_config: &CompactionConfig) -> bool {
        l0_size
            > compaction_config
                .level0_stop_write_threshold_max_size
                .unwrap_or(compaction_config::level0_stop_write_threshold_max_size())
    }

    pub fn write_stop_l0_file_count(
        l0_file_count: usize,
        compaction_config: &CompactionConfig,
    ) -> bool {
        l0_file_count
            > compaction_config
                .level0_stop_write_threshold_max_sst_count
                .unwrap_or(compaction_config::level0_stop_write_threshold_max_sst_count())
                as usize
    }

    pub fn emergency_l0_file_count(
        l0_file_count: usize,
        compaction_config: &CompactionConfig,
    ) -> bool {
        l0_file_count
            > compaction_config
                .emergency_level0_sst_file_count
                .unwrap_or(compaction_config::emergency_level0_sst_file_count())
                as usize
    }

    pub fn emergency_l0_partition_count(
        last_l0_sub_level_partition_count: usize,
        compaction_config: &CompactionConfig,
    ) -> bool {
        last_l0_sub_level_partition_count
            > compaction_config
                .emergency_level0_sub_level_partition
                .unwrap_or(compaction_config::emergency_level0_sub_level_partition())
                as usize
    }

    pub fn check_single_group_write_stop(
        levels: &Levels,
        compaction_config: &CompactionConfig,
    ) -> GroupState {
        if Self::write_stop_sub_level_count(levels.l0.sub_levels.len(), compaction_config) {
            return GroupState::WriteStop(format!(
                "WriteStop(l0_level_count: {}, threshold: {}) too many L0 sub levels",
                levels.l0.sub_levels.len(),
                compaction_config.level0_stop_write_threshold_sub_level_number
            ));
        }

        if Self::write_stop_l0_file_count(
            levels
                .l0
                .sub_levels
                .iter()
                .map(|l| l.table_infos.len())
                .sum(),
            compaction_config,
        ) {
            return GroupState::WriteStop(format!(
                "WriteStop(l0_sst_count: {}, threshold: {}) too many L0 sst files",
                levels
                    .l0
                    .sub_levels
                    .iter()
                    .map(|l| l.table_infos.len())
                    .sum::<usize>(),
                compaction_config
                    .level0_stop_write_threshold_max_sst_count
                    .unwrap_or(compaction_config::level0_stop_write_threshold_max_sst_count())
            ));
        }

        if Self::write_stop_l0_size(levels.l0.total_file_size, compaction_config) {
            return GroupState::WriteStop(format!(
                "WriteStop(l0_size: {}, threshold: {}) too large L0 size",
                levels.l0.total_file_size,
                compaction_config
                    .level0_stop_write_threshold_max_size
                    .unwrap_or(compaction_config::level0_stop_write_threshold_max_size())
            ));
        }

        GroupState::Normal
    }

    pub fn check_single_group_emergency(
        levels: &Levels,
        compaction_config: &CompactionConfig,
    ) -> GroupState {
        if Self::emergency_l0_file_count(
            levels
                .l0
                .sub_levels
                .iter()
                .map(|l| l.table_infos.len())
                .sum(),
            compaction_config,
        ) {
            return GroupState::Emergency(format!(
                "Emergency(l0_sst_count: {}, threshold: {}) too many L0 sst files",
                levels
                    .l0
                    .sub_levels
                    .iter()
                    .map(|l| l.table_infos.len())
                    .sum::<usize>(),
                compaction_config
                    .emergency_level0_sst_file_count
                    .unwrap_or(compaction_config::emergency_level0_sst_file_count())
            ));
        }

        if Self::emergency_l0_partition_count(
            levels
                .l0
                .sub_levels
                .first()
                .map(|l| l.table_infos.len())
                .unwrap_or(0),
            compaction_config,
        ) {
            return GroupState::Emergency(format!(
                "Emergency(l0_partition_count: {}, threshold: {}) too many L0 partitions",
                levels
                    .l0
                    .sub_levels
                    .first()
                    .map(|l| l.table_infos.len())
                    .unwrap_or(0),
                compaction_config
                    .emergency_level0_sub_level_partition
                    .unwrap_or(compaction_config::emergency_level0_sub_level_partition())
            ));
        }

        GroupState::Normal
    }

    pub fn group_state(levels: &Levels, compaction_config: &CompactionConfig) -> GroupState {
        let state = Self::check_single_group_write_stop(levels, compaction_config);
        if state.is_write_stop() {
            return state;
        }

        Self::check_single_group_emergency(levels, compaction_config)
    }
}
