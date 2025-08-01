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

use std::collections::HashSet;

use either::Either;
use pgwire::pg_response::{PgResponse, StatementType};
use risingwave_common::catalog::{FunctionId, ObjectId, TableId};
use risingwave_pb::serverless_backfill_controller::{
    ProvisionRequest, node_group_controller_service_client,
};
use risingwave_pb::stream_plan::PbStreamFragmentGraph;
use risingwave_sqlparser::ast::{EmitMode, Ident, ObjectName, Query};
use thiserror_ext::AsReport;

use super::RwPgResponse;
use crate::binder::{Binder, BoundQuery, BoundSetExpr};
use crate::catalog::check_column_name_not_reserved;
use crate::error::ErrorCode::{InvalidInputSyntax, ProtocolError};
use crate::error::{ErrorCode, Result, RwError};
use crate::handler::HandlerArgs;
use crate::optimizer::backfill_order_strategy::plan_backfill_order;
use crate::optimizer::plan_node::generic::GenericPlanRef;
use crate::optimizer::plan_node::{Explain, Stream};
use crate::optimizer::{OptimizerContext, OptimizerContextRef, PlanRef, RelationCollectorVisitor};
use crate::planner::Planner;
use crate::scheduler::streaming_manager::CreatingStreamingJobInfo;
use crate::session::{SESSION_MANAGER, SessionImpl};
use crate::stream_fragmenter::{GraphJobType, build_graph_with_strategy};
use crate::utils::ordinal;
use crate::{TableCatalog, WithOptions};

pub const RESOURCE_GROUP_KEY: &str = "resource_group";
pub const CLOUD_SERVERLESS_BACKFILL_ENABLED: &str = "cloud.serverless_backfill_enabled";

pub(super) fn parse_column_names(columns: &[Ident]) -> Option<Vec<String>> {
    if columns.is_empty() {
        None
    } else {
        Some(columns.iter().map(|v| v.real_value()).collect())
    }
}

/// If columns is empty, it means that the user did not specify the column names.
/// In this case, we extract the column names from the query.
/// If columns is not empty, it means that user specify the column names and the user
/// should guarantee that the column names number are consistent with the query.
pub(super) fn get_column_names(
    bound: &BoundQuery,
    columns: Vec<Ident>,
) -> Result<Option<Vec<String>>> {
    let col_names = parse_column_names(&columns);
    if let BoundSetExpr::Select(select) = &bound.body {
        // `InputRef`'s alias will be implicitly assigned in `bind_project`.
        // If user provides columns name (col_names.is_some()), we don't need alias.
        // For other expressions (col_names.is_none()), we require the user to explicitly assign an
        // alias.
        if col_names.is_none() {
            for (i, alias) in select.aliases.iter().enumerate() {
                if alias.is_none() {
                    return Err(ErrorCode::BindError(format!(
                    "An alias must be specified for the {} expression (counting from 1) in result relation", ordinal(i+1)
                ))
                .into());
                }
            }
        }
    }

    Ok(col_names)
}

/// Bind and generate create MV plan, return plan and mv table info.
pub fn gen_create_mv_plan(
    session: &SessionImpl,
    context: OptimizerContextRef,
    query: Query,
    name: ObjectName,
    columns: Vec<Ident>,
    emit_mode: Option<EmitMode>,
) -> Result<(PlanRef, TableCatalog)> {
    let mut binder = Binder::new_for_stream(session);
    let bound = binder.bind_query(&query)?;
    gen_create_mv_plan_bound(session, context, bound, name, columns, emit_mode)
}

/// Generate create MV plan from a bound query
pub fn gen_create_mv_plan_bound(
    session: &SessionImpl,
    context: OptimizerContextRef,
    query: BoundQuery,
    name: ObjectName,
    columns: Vec<Ident>,
    emit_mode: Option<EmitMode>,
) -> Result<(PlanRef, TableCatalog)> {
    if session.config().create_compaction_group_for_mv() {
        context.warn_to_user("The session variable CREATE_COMPACTION_GROUP_FOR_MV has been deprecated. It will not take effect.");
    }

    let db_name = &session.database();
    let (schema_name, table_name) = Binder::resolve_schema_qualified_name(db_name, &name)?;

    let (database_id, schema_id) = session.get_database_and_schema_id_for_create(schema_name)?;

    let definition = context.normalized_sql().to_owned();

    let col_names = get_column_names(&query, columns)?;

    let emit_on_window_close = emit_mode == Some(EmitMode::OnWindowClose);
    if emit_on_window_close {
        context.warn_to_user("EMIT ON WINDOW CLOSE is currently an experimental feature. Please use it with caution.");
    }

    let mut plan_root = Planner::new_for_stream(context).plan_query(query)?;
    if let Some(col_names) = col_names {
        for name in &col_names {
            check_column_name_not_reserved(name)?;
        }
        plan_root.set_out_names(col_names)?;
    }
    let materialize = plan_root.gen_materialize_plan(
        database_id,
        schema_id,
        table_name,
        definition,
        emit_on_window_close,
    )?;

    let mut table = materialize.table().clone();
    table.owner = session.user_id();

    let plan: PlanRef = materialize.into();

    let ctx = plan.ctx();
    let explain_trace = ctx.is_explain_trace();
    if explain_trace {
        ctx.trace("Create Materialized View:");
        ctx.trace(plan.explain_to_string());
    }

    Ok((plan, table))
}

pub async fn handle_create_mv(
    handler_args: HandlerArgs,
    if_not_exists: bool,
    name: ObjectName,
    query: Query,
    columns: Vec<Ident>,
    emit_mode: Option<EmitMode>,
) -> Result<RwPgResponse> {
    let (dependent_relations, dependent_udfs, bound_query) = {
        let mut binder = Binder::new_for_stream(handler_args.session.as_ref());
        let bound_query = binder.bind_query(&query)?;
        (
            binder.included_relations().clone(),
            binder.included_udfs().clone(),
            bound_query,
        )
    };
    handle_create_mv_bound(
        handler_args,
        if_not_exists,
        name,
        bound_query,
        dependent_relations,
        dependent_udfs,
        columns,
        emit_mode,
    )
    .await
}

/// Send a provision request to the serverless backfill controller
pub async fn provision_resource_group(sbc_addr: String) -> Result<String> {
    let request = tonic::Request::new(ProvisionRequest {});
    let mut client =
        node_group_controller_service_client::NodeGroupControllerServiceClient::connect(
            sbc_addr.clone(),
        )
        .await
        .map_err(|e| {
            RwError::from(ErrorCode::InternalError(format!(
                "unable to reach serverless backfill controller at addr {}: {}",
                sbc_addr,
                e.as_report()
            )))
        })?;

    match client.provision(request).await {
        Ok(resp) => Ok(resp.into_inner().resource_group),
        Err(e) => Err(RwError::from(ErrorCode::InternalError(format!(
            "serverless backfill controller returned error :{}",
            e.as_report()
        )))),
    }
}

fn get_with_options(handler_args: HandlerArgs) -> WithOptions {
    let context = OptimizerContext::from_handler_args(handler_args);
    context.with_options().clone()
}

pub async fn handle_create_mv_bound(
    handler_args: HandlerArgs,
    if_not_exists: bool,
    name: ObjectName,
    query: BoundQuery,
    dependent_relations: HashSet<TableId>,
    dependent_udfs: HashSet<FunctionId>, // TODO(rc): merge with `dependent_relations`
    columns: Vec<Ident>,
    emit_mode: Option<EmitMode>,
) -> Result<RwPgResponse> {
    let session = handler_args.session.clone();

    // Check cluster limits
    session.check_cluster_limits().await?;

    if let Either::Right(resp) = session.check_relation_name_duplicated(
        name.clone(),
        StatementType::CREATE_MATERIALIZED_VIEW,
        if_not_exists,
    )? {
        return Ok(resp);
    }

    let (table, graph, dependencies, resource_group) = {
        gen_create_mv_graph(
            handler_args,
            name,
            query,
            dependent_relations,
            dependent_udfs,
            columns,
            emit_mode,
        )
        .await?
    };

    // Ensure writes to `StreamJobTracker` are atomic.
    let _job_guard =
        session
            .env()
            .creating_streaming_job_tracker()
            .guard(CreatingStreamingJobInfo::new(
                session.session_id(),
                table.database_id,
                table.schema_id,
                table.name.clone(),
            ));

    let catalog_writer = session.catalog_writer()?;
    catalog_writer
        .create_materialized_view(
            table.to_prost(),
            graph,
            dependencies,
            resource_group,
            if_not_exists,
        )
        .await?;

    Ok(PgResponse::empty_result(
        StatementType::CREATE_MATERIALIZED_VIEW,
    ))
}

pub(crate) async fn gen_create_mv_graph(
    handler_args: HandlerArgs,
    name: ObjectName,
    query: BoundQuery,
    dependent_relations: HashSet<TableId>,
    dependent_udfs: HashSet<FunctionId>,
    columns: Vec<Ident>,
    emit_mode: Option<EmitMode>,
) -> Result<(
    TableCatalog,
    PbStreamFragmentGraph,
    HashSet<u32>,
    Option<String>,
)> {
    let mut with_options = get_with_options(handler_args.clone());
    let mut resource_group = with_options.remove(&RESOURCE_GROUP_KEY.to_owned());

    if resource_group.is_some() {
        risingwave_common::license::Feature::ResourceGroup
            .check_available()
            .map_err(|e| anyhow::anyhow!(e))?;
    }

    let is_serverless_backfill = with_options
        .remove(&CLOUD_SERVERLESS_BACKFILL_ENABLED.to_owned())
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false);

    if resource_group.is_some() && is_serverless_backfill {
        return Err(RwError::from(InvalidInputSyntax(
            "Please do not specify serverless backfilling and resource group together".to_owned(),
        )));
    }

    if !with_options.is_empty() {
        // get other useful fields by `remove`, the logic here is to reject unknown options.
        return Err(RwError::from(ProtocolError(format!(
            "unexpected options in WITH clause: {:?}",
            with_options.keys()
        ))));
    }

    let sbc_addr = match SESSION_MANAGER.get() {
        Some(manager) => manager.env().sbc_address(),
        None => "",
    }
    .to_owned();

    if is_serverless_backfill && sbc_addr.is_empty() {
        return Err(RwError::from(InvalidInputSyntax(
            "Serverless Backfill is disabled on-premise. Use RisingWave cloud at https://cloud.risingwave.com/auth/signup to try this feature".to_owned(),
        )));
    }

    if is_serverless_backfill {
        match provision_resource_group(sbc_addr).await {
            Err(e) => {
                return Err(RwError::from(ProtocolError(format!(
                    "failed to provision serverless backfill nodes: {}",
                    e.as_report()
                ))));
            }
            Ok(val) => resource_group = Some(val),
        }
    }
    tracing::debug!(
        resource_group = resource_group,
        "provisioning on resource group"
    );

    let context = OptimizerContext::from_handler_args(handler_args);
    let has_order_by = !query.order.is_empty();
    if has_order_by {
        context.warn_to_user(r#"The ORDER BY clause in the CREATE MATERIALIZED VIEW statement does not guarantee that the rows selected out of this materialized view is returned in this order.
It only indicates the physical clustering of the data, which may improve the performance of queries issued against this materialized view.
"#.to_owned());
    }

    if resource_group.is_some()
        && !context
            .session_ctx()
            .config()
            .streaming_use_arrangement_backfill()
    {
        return Err(RwError::from(ProtocolError("The session config arrangement backfill must be enabled to use the resource_group option".to_owned())));
    }

    let context: OptimizerContextRef = context.into();
    let session = context.session_ctx().as_ref();

    let (plan, table) =
        gen_create_mv_plan_bound(session, context.clone(), query, name, columns, emit_mode)?;

    let backfill_order = plan_backfill_order(
        session,
        context.with_options().backfill_order_strategy(),
        plan.clone(),
    )?;

    // TODO(rc): To be consistent with UDF dependency check, we should collect relation dependencies
    // during binding instead of visiting the optimized plan.
    let dependencies =
        RelationCollectorVisitor::collect_with::<Stream>(dependent_relations, plan.clone())
            .into_iter()
            .map(|id| id.table_id() as ObjectId)
            .chain(
                dependent_udfs
                    .into_iter()
                    .map(|id| id.function_id() as ObjectId),
            )
            .collect();

    let graph = build_graph_with_strategy(
        plan,
        Some(GraphJobType::MaterializedView),
        Some(backfill_order),
    )?;

    Ok((table, graph, dependencies, resource_group))
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashMap;

    use pgwire::pg_response::StatementType::CREATE_MATERIALIZED_VIEW;
    use risingwave_common::catalog::{
        DEFAULT_DATABASE_NAME, DEFAULT_SCHEMA_NAME, ROW_ID_COLUMN_NAME, RW_TIMESTAMP_COLUMN_NAME,
    };
    use risingwave_common::types::{DataType, StructType};

    use crate::catalog::root_catalog::SchemaPath;
    use crate::test_utils::{LocalFrontend, PROTO_FILE_DATA, create_proto_file};

    #[tokio::test]
    async fn test_create_mv_handler() {
        let proto_file = create_proto_file(PROTO_FILE_DATA);
        let sql = format!(
            r#"CREATE SOURCE t1
    WITH (connector = 'kinesis')
    FORMAT PLAIN ENCODE PROTOBUF (message = '.test.TestRecord', schema.location = 'file://{}')"#,
            proto_file.path().to_str().unwrap()
        );
        let frontend = LocalFrontend::new(Default::default()).await;
        frontend.run_sql(sql).await.unwrap();

        let sql = "create materialized view mv1 as select t1.country from t1";
        frontend.run_sql(sql).await.unwrap();

        let session = frontend.session_ref();
        let catalog_reader = session.env().catalog_reader().read_guard();
        let schema_path = SchemaPath::Name(DEFAULT_SCHEMA_NAME);

        // Check source exists.
        let (source, _) = catalog_reader
            .get_source_by_name(DEFAULT_DATABASE_NAME, schema_path, "t1")
            .unwrap();
        assert_eq!(source.name, "t1");

        // Check table exists.
        let (table, _) = catalog_reader
            .get_created_table_by_name(DEFAULT_DATABASE_NAME, schema_path, "mv1")
            .unwrap();
        assert_eq!(table.name(), "mv1");

        let columns = table
            .columns
            .iter()
            .map(|col| (col.name(), col.data_type().clone()))
            .collect::<HashMap<&str, DataType>>();

        let city_type = StructType::new(vec![
            ("address", DataType::Varchar),
            ("zipcode", DataType::Varchar),
        ])
        // .with_ids([5, 6].map(ColumnId::new))
        .into();
        let expected_columns = maplit::hashmap! {
            ROW_ID_COLUMN_NAME => DataType::Serial,
            "country" => StructType::new(
                 vec![("address", DataType::Varchar),("city", city_type),("zipcode", DataType::Varchar)],
            )
            // .with_ids([3, 4, 7].map(ColumnId::new))
            .into(),
            RW_TIMESTAMP_COLUMN_NAME => DataType::Timestamptz,
        };
        assert_eq!(columns, expected_columns, "{columns:#?}");
    }

    /// When creating MV, a unique column name must be specified for each column
    #[tokio::test]
    async fn test_no_alias() {
        let frontend = LocalFrontend::new(Default::default()).await;

        let sql = "create table t(x varchar)";
        frontend.run_sql(sql).await.unwrap();

        // Aggregation without alias is ok.
        let sql = "create materialized view mv0 as select count(x) from t";
        frontend.run_sql(sql).await.unwrap();

        // Same aggregations without alias is forbidden, because it make the same column name.
        let sql = "create materialized view mv1 as select count(x), count(*) from t";
        let err = frontend.run_sql(sql).await.unwrap_err();
        assert_eq!(
            err.to_string(),
            "Invalid input syntax: column \"count\" specified more than once"
        );

        // Literal without alias is forbidden.
        let sql = "create materialized view mv1 as select 1";
        let err = frontend.run_sql(sql).await.unwrap_err();
        assert_eq!(
            err.to_string(),
            "Bind error: An alias must be specified for the 1st expression (counting from 1) in result relation"
        );

        // some expression without alias is forbidden.
        let sql = "create materialized view mv1 as select x is null from t";
        let err = frontend.run_sql(sql).await.unwrap_err();
        assert_eq!(
            err.to_string(),
            "Bind error: An alias must be specified for the 1st expression (counting from 1) in result relation"
        );
    }

    /// Creating MV with order by returns a special notice
    #[tokio::test]
    async fn test_create_mv_with_order_by() {
        let frontend = LocalFrontend::new(Default::default()).await;

        let sql = "create table t(x varchar)";
        frontend.run_sql(sql).await.unwrap();

        // Without order by
        let sql = "create materialized view mv1 as select * from t";
        let response = frontend.run_sql(sql).await.unwrap();
        assert_eq!(response.stmt_type(), CREATE_MATERIALIZED_VIEW);
        assert!(response.notices().is_empty());

        // With order by
        let sql = "create materialized view mv2 as select * from t order by x";
        let response = frontend.run_sql(sql).await.unwrap();
        assert_eq!(response.stmt_type(), CREATE_MATERIALIZED_VIEW);
    }
}
