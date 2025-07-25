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

pub mod compaction;
mod jni_catalog;
mod mock_catalog;
mod storage_catalog;

use std::collections::HashMap;
use std::sync::Arc;

use ::iceberg::io::{S3_ACCESS_KEY_ID, S3_ENDPOINT, S3_REGION, S3_SECRET_ACCESS_KEY};
use ::iceberg::table::Table;
use ::iceberg::{Catalog, TableIdent};
use anyhow::{Context, anyhow};
use iceberg::io::{
    AZBLOB_ACCOUNT_KEY, AZBLOB_ACCOUNT_NAME, AZBLOB_ENDPOINT, GCS_CREDENTIALS_JSON,
    GCS_DISABLE_CONFIG_LOAD, S3_DISABLE_CONFIG_LOAD,
};
use iceberg_catalog_glue::{AWS_ACCESS_KEY_ID, AWS_REGION_NAME, AWS_SECRET_ACCESS_KEY};
use phf::{Set, phf_set};
use risingwave_common::bail;
use risingwave_common::util::env_var::env_var_is_true;
use serde_derive::Deserialize;
use serde_with::serde_as;
use url::Url;
use with_options::WithOptions;

use crate::connector_common::common::DISABLE_DEFAULT_CREDENTIAL;
use crate::connector_common::iceberg::storage_catalog::StorageCatalogConfig;
use crate::deserialize_optional_bool_from_string;
use crate::enforce_secret::EnforceSecret;
use crate::error::ConnectorResult;

#[serde_as]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, WithOptions)]
pub struct IcebergCommon {
    // Catalog type supported by iceberg, such as "storage", "rest".
    // If not set, we use "storage" as default.
    #[serde(rename = "catalog.type")]
    pub catalog_type: Option<String>,
    #[serde(rename = "s3.region")]
    pub region: Option<String>,
    #[serde(rename = "s3.endpoint")]
    pub endpoint: Option<String>,
    #[serde(rename = "s3.access.key")]
    pub access_key: Option<String>,
    #[serde(rename = "s3.secret.key")]
    pub secret_key: Option<String>,

    #[serde(rename = "gcs.credential")]
    pub gcs_credential: Option<String>,

    #[serde(rename = "azblob.account_name")]
    pub azblob_account_name: Option<String>,
    #[serde(rename = "azblob.account_key")]
    pub azblob_account_key: Option<String>,
    #[serde(rename = "azblob.endpoint_url")]
    pub azblob_endpoint_url: Option<String>,

    /// Path of iceberg warehouse.
    #[serde(rename = "warehouse.path")]
    pub warehouse_path: Option<String>,
    /// AWS Client id, can be omitted for storage catalog or when
    /// caller's AWS account ID matches glue id
    #[serde(rename = "glue.id")]
    pub glue_id: Option<String>,
    /// Catalog name, default value is risingwave.
    #[serde(rename = "catalog.name")]
    pub catalog_name: Option<String>,
    /// URI of iceberg catalog, only applicable in rest catalog.
    #[serde(rename = "catalog.uri")]
    pub catalog_uri: Option<String>,
    #[serde(rename = "database.name")]
    pub database_name: Option<String>,
    /// Full name of table, must include schema name.
    #[serde(rename = "table.name")]
    pub table_name: String,
    /// Credential for accessing iceberg catalog, only applicable in rest catalog.
    /// A credential to exchange for a token in the `OAuth2` client credentials flow.
    #[serde(rename = "catalog.credential")]
    pub credential: Option<String>,
    /// token for accessing iceberg catalog, only applicable in rest catalog.
    /// A Bearer token which will be used for interaction with the server.
    #[serde(rename = "catalog.token")]
    pub token: Option<String>,
    /// `oauth2_server_uri` for accessing iceberg catalog, only applicable in rest catalog.
    /// Token endpoint URI to fetch token from if the Rest Catalog is not the authorization server.
    #[serde(rename = "catalog.oauth2_server_uri")]
    pub oauth2_server_uri: Option<String>,
    /// scope for accessing iceberg catalog, only applicable in rest catalog.
    /// Additional scope for `OAuth2`.
    #[serde(rename = "catalog.scope")]
    pub scope: Option<String>,

    /// The signing region to use when signing requests to the REST catalog.
    #[serde(rename = "catalog.rest.signing_region")]
    pub rest_signing_region: Option<String>,

    /// The signing name to use when signing requests to the REST catalog.
    #[serde(rename = "catalog.rest.signing_name")]
    pub rest_signing_name: Option<String>,

    /// Whether to use `SigV4` for signing requests to the REST catalog.
    #[serde(
        rename = "catalog.rest.sigv4_enabled",
        default,
        deserialize_with = "deserialize_optional_bool_from_string"
    )]
    pub rest_sigv4_enabled: Option<bool>,

    #[serde(
        rename = "s3.path.style.access",
        default,
        deserialize_with = "deserialize_optional_bool_from_string"
    )]
    pub path_style_access: Option<bool>,
    /// Enable config load. This parameter set to true will load warehouse credentials from the environment. Only allowed to be used in a self-hosted environment.
    #[serde(default, deserialize_with = "deserialize_optional_bool_from_string")]
    pub enable_config_load: Option<bool>,

    /// This is only used by iceberg engine to enable the hosted catalog.
    #[serde(
        rename = "hosted_catalog",
        default,
        deserialize_with = "deserialize_optional_bool_from_string"
    )]
    pub hosted_catalog: Option<bool>,

    /// The http header to be used in the catalog requests.
    /// Example:
    /// `catalog.header = "key1=value1;key2=value2;key3=value3"`
    /// explain the format of the header:
    /// - Each header is a key-value pair, separated by an '='.
    /// - Multiple headers can be specified, separated by a ';'.
    #[serde(rename = "catalog.header")]
    pub header: Option<String>,
}

impl EnforceSecret for IcebergCommon {
    const ENFORCE_SECRET_PROPERTIES: Set<&'static str> = phf_set! {
        "s3.access.key",
        "s3.secret.key",
        "gcs.credential",
        "catalog.credential",
        "catalog.token",
        "catalog.oauth2_server_uri",
    };
}

impl IcebergCommon {
    pub fn catalog_type(&self) -> &str {
        self.catalog_type.as_deref().unwrap_or("storage")
    }

    pub fn catalog_name(&self) -> String {
        self.catalog_name
            .as_ref()
            .cloned()
            .unwrap_or_else(|| "risingwave".to_owned())
    }

    pub fn headers(&self) -> ConnectorResult<HashMap<String, String>> {
        if let Some(header) = &self.header {
            let mut headers = HashMap::new();
            for pair in header.split(';') {
                let mut parts = pair.split('=');
                if let (Some(key), Some(value)) = (parts.next(), parts.next()) {
                    headers.insert(key.to_owned(), value.to_owned());
                } else {
                    bail!("Invalid header format: {}", pair);
                }
            }
            Ok(headers)
        } else {
            Ok(HashMap::new())
        }
    }

    pub fn enable_config_load(&self) -> bool {
        // If the env var is set to true, we disable the default config load. (Cloud environment)
        if env_var_is_true(DISABLE_DEFAULT_CREDENTIAL) {
            return false;
        }
        self.enable_config_load.unwrap_or(false)
    }

    /// For both V1 and V2.
    fn build_jni_catalog_configs(
        &self,
        java_catalog_props: &HashMap<String, String>,
    ) -> ConnectorResult<(HashMap<String, String>, HashMap<String, String>)> {
        let mut iceberg_configs = HashMap::new();
        let enable_config_load = self.enable_config_load();
        let file_io_props = {
            let catalog_type = self.catalog_type().to_owned();

            if let Some(region) = &self.region {
                // iceberg-rust
                iceberg_configs.insert(S3_REGION.to_owned(), region.clone());
            }

            if let Some(endpoint) = &self.endpoint {
                // iceberg-rust
                iceberg_configs.insert(S3_ENDPOINT.to_owned(), endpoint.clone());
            }

            // iceberg-rust
            if let Some(access_key) = &self.access_key {
                iceberg_configs.insert(S3_ACCESS_KEY_ID.to_owned(), access_key.clone());
            }
            if let Some(secret_key) = &self.secret_key {
                iceberg_configs.insert(S3_SECRET_ACCESS_KEY.to_owned(), secret_key.clone());
            }
            if let Some(gcs_credential) = &self.gcs_credential {
                iceberg_configs.insert(GCS_CREDENTIALS_JSON.to_owned(), gcs_credential.clone());
                if catalog_type != "rest" && catalog_type != "rest_rust" {
                    bail!("gcs unsupported in {} catalog", &catalog_type);
                }
            }

            if let (
                Some(azblob_account_name),
                Some(azblob_account_key),
                Some(azblob_endpoint_url),
            ) = (
                &self.azblob_account_name,
                &self.azblob_account_key,
                &self.azblob_endpoint_url,
            ) {
                iceberg_configs.insert(AZBLOB_ACCOUNT_NAME.to_owned(), azblob_account_name.clone());
                iceberg_configs.insert(AZBLOB_ACCOUNT_KEY.to_owned(), azblob_account_key.clone());
                iceberg_configs.insert(AZBLOB_ENDPOINT.to_owned(), azblob_endpoint_url.clone());

                if catalog_type != "rest" && catalog_type != "rest_rust" {
                    bail!("azblob unsupported in {} catalog", &catalog_type);
                }
            }

            match &self.warehouse_path {
                Some(warehouse_path) => {
                    let (bucket, _) = {
                        let is_s3_tables = warehouse_path.starts_with("arn:aws:s3tables");
                        let url = Url::parse(warehouse_path);
                        if (url.is_err() || is_s3_tables)
                            && (catalog_type == "rest" || catalog_type == "rest_rust")
                        {
                            // If the warehouse path is not a valid URL, it could be a warehouse name in rest catalog
                            // Or it could be a s3tables path, which is not a valid URL but a valid warehouse path,
                            // so we allow it to pass here.
                            (None, None)
                        } else {
                            let url = url.with_context(|| {
                                format!("Invalid warehouse path: {}", warehouse_path)
                            })?;
                            let bucket = url
                                .host_str()
                                .with_context(|| {
                                    format!(
                                        "Invalid s3 path: {}, bucket is missing",
                                        warehouse_path
                                    )
                                })?
                                .to_owned();
                            let root = url.path().trim_start_matches('/').to_owned();
                            (Some(bucket), Some(root))
                        }
                    };

                    if let Some(bucket) = bucket {
                        iceberg_configs.insert("iceberg.table.io.bucket".to_owned(), bucket);
                    }
                }
                None => {
                    if catalog_type != "rest" && catalog_type != "rest_rust" {
                        bail!("`warehouse.path` must be set in {} catalog", &catalog_type);
                    }
                }
            }
            iceberg_configs.insert(
                S3_DISABLE_CONFIG_LOAD.to_owned(),
                (!enable_config_load).to_string(),
            );

            iceberg_configs.insert(
                GCS_DISABLE_CONFIG_LOAD.to_owned(),
                (!enable_config_load).to_string(),
            );

            iceberg_configs
        };

        // Prepare jni configs, for details please see https://iceberg.apache.org/docs/latest/aws/
        let mut java_catalog_configs = HashMap::new();
        {
            if let Some(uri) = self.catalog_uri.as_deref() {
                java_catalog_configs.insert("uri".to_owned(), uri.to_owned());
            }

            if let Some(warehouse_path) = &self.warehouse_path {
                java_catalog_configs.insert("warehouse".to_owned(), warehouse_path.clone());
            }
            java_catalog_configs.extend(java_catalog_props.clone());

            // Currently we only support s3, so let's set it to s3
            java_catalog_configs.insert(
                "io-impl".to_owned(),
                "org.apache.iceberg.aws.s3.S3FileIO".to_owned(),
            );

            // suppress log of S3FileIO like: Unclosed S3FileIO instance created by...
            java_catalog_configs.insert("init-creation-stacktrace".to_owned(), "false".to_owned());

            if let Some(region) = &self.region {
                java_catalog_configs.insert("client.region".to_owned(), region.clone());
            }
            if let Some(endpoint) = &self.endpoint {
                java_catalog_configs.insert("s3.endpoint".to_owned(), endpoint.clone());
            }

            if let Some(access_key) = &self.access_key {
                java_catalog_configs.insert("s3.access-key-id".to_owned(), access_key.clone());
            }
            if let Some(secret_key) = &self.secret_key {
                java_catalog_configs.insert("s3.secret-access-key".to_owned(), secret_key.clone());
            }

            if let Some(path_style_access) = self.path_style_access {
                java_catalog_configs.insert(
                    "s3.path-style-access".to_owned(),
                    path_style_access.to_string(),
                );
            }

            let headers = self.headers()?;
            for (header_name, header_value) in headers {
                java_catalog_configs.insert(format!("header.{}", header_name), header_value);
            }

            match self.catalog_type.as_deref() {
                Some("rest") => {
                    if let Some(credential) = &self.credential {
                        java_catalog_configs.insert("credential".to_owned(), credential.clone());
                    }
                    if let Some(token) = &self.token {
                        java_catalog_configs.insert("token".to_owned(), token.clone());
                    }
                    if let Some(oauth2_server_uri) = &self.oauth2_server_uri {
                        java_catalog_configs
                            .insert("oauth2-server-uri".to_owned(), oauth2_server_uri.clone());
                    }
                    if let Some(scope) = &self.scope {
                        java_catalog_configs.insert("scope".to_owned(), scope.clone());
                    }
                    if let Some(rest_signing_region) = &self.rest_signing_region {
                        java_catalog_configs.insert(
                            "rest.signing-region".to_owned(),
                            rest_signing_region.clone(),
                        );
                    }
                    if let Some(rest_signing_name) = &self.rest_signing_name {
                        java_catalog_configs
                            .insert("rest.signing-name".to_owned(), rest_signing_name.clone());
                    }
                    if let Some(rest_sigv4_enabled) = self.rest_sigv4_enabled {
                        java_catalog_configs.insert(
                            "rest.sigv4-enabled".to_owned(),
                            rest_sigv4_enabled.to_string(),
                        );

                        if let Some(access_key) = &self.access_key {
                            java_catalog_configs
                                .insert("rest.access-key-id".to_owned(), access_key.clone());
                        }

                        if let Some(secret_key) = &self.secret_key {
                            java_catalog_configs
                                .insert("rest.secret-access-key".to_owned(), secret_key.clone());
                        }
                    }
                }
                Some("glue") => {
                    if !enable_config_load {
                        java_catalog_configs.insert(
                            "client.credentials-provider".to_owned(),
                            "com.risingwave.connector.catalog.GlueCredentialProvider".to_owned(),
                        );
                        // Use S3 ak/sk and region as glue ak/sk and region by default.
                        // TODO: use different ak/sk and region for s3 and glue.
                        if let Some(access_key) = &self.access_key {
                            java_catalog_configs.insert(
                                "client.credentials-provider.glue.access-key-id".to_owned(),
                                access_key.clone(),
                            );
                        }
                        if let Some(secret_key) = &self.secret_key {
                            java_catalog_configs.insert(
                                "client.credentials-provider.glue.secret-access-key".to_owned(),
                                secret_key.clone(),
                            );
                        }
                    }

                    if let Some(region) = &self.region {
                        java_catalog_configs.insert("client.region".to_owned(), region.clone());
                        java_catalog_configs.insert(
                            "glue.endpoint".to_owned(),
                            format!("https://glue.{}.amazonaws.com", region),
                        );
                    }

                    if let Some(glue_id) = self.glue_id.as_deref() {
                        java_catalog_configs.insert("glue.id".to_owned(), glue_id.to_owned());
                    }
                }
                _ => {}
            }
        }

        Ok((file_io_props, java_catalog_configs))
    }
}

impl IcebergCommon {
    pub fn full_table_name(&self) -> ConnectorResult<TableIdent> {
        let ret = if let Some(database_name) = &self.database_name {
            TableIdent::from_strs(vec![database_name, &self.table_name])
        } else {
            TableIdent::from_strs(vec![&self.table_name])
        };

        Ok(ret.context("Failed to create table identifier")?)
    }

    /// TODO: remove the arguments and put them into `IcebergCommon`. Currently the handling in source and sink are different, so pass them separately to be safer.
    pub async fn create_catalog(
        &self,
        java_catalog_props: &HashMap<String, String>,
    ) -> ConnectorResult<Arc<dyn Catalog>> {
        match self.catalog_type() {
            "storage" => {
                let warehouse = self
                    .warehouse_path
                    .clone()
                    .ok_or_else(|| anyhow!("`warehouse.path` must be set in storage catalog"))?;
                let url = Url::parse(warehouse.as_ref())
                    .map_err(|_| anyhow!("Invalid warehouse path: {}", warehouse))?;

                let config = match url.scheme() {
                    "s3" | "s3a" => StorageCatalogConfig::S3(
                        storage_catalog::StorageCatalogS3Config::builder()
                            .warehouse(warehouse)
                            .access_key(self.access_key.clone())
                            .secret_key(self.secret_key.clone())
                            .region(self.region.clone())
                            .endpoint(self.endpoint.clone())
                            .enable_config_load(Some(self.enable_config_load()))
                            .build(),
                    ),
                    "gs" | "gcs" => StorageCatalogConfig::Gcs(
                        storage_catalog::StorageCatalogGcsConfig::builder()
                            .warehouse(warehouse)
                            .credential(self.gcs_credential.clone())
                            .enable_config_load(Some(self.enable_config_load()))
                            .build(),
                    ),
                    "azblob" => StorageCatalogConfig::Azblob(
                        storage_catalog::StorageCatalogAzblobConfig::builder()
                            .warehouse(warehouse)
                            .account_name(self.azblob_account_name.clone())
                            .account_key(self.azblob_account_key.clone())
                            .endpoint(self.azblob_endpoint_url.clone())
                            .build(),
                    ),
                    scheme => bail!("Unsupported warehouse scheme: {}", scheme),
                };

                let catalog = storage_catalog::StorageCatalog::new(config)?;
                Ok(Arc::new(catalog))
            }
            "rest_rust" => {
                let mut iceberg_configs = HashMap::new();

                // check gcs credential or s3 access key and secret key
                if let Some(gcs_credential) = &self.gcs_credential {
                    iceberg_configs.insert(GCS_CREDENTIALS_JSON.to_owned(), gcs_credential.clone());
                } else {
                    if let Some(region) = &self.region {
                        iceberg_configs.insert(S3_REGION.to_owned(), region.clone());
                    }
                    if let Some(endpoint) = &self.endpoint {
                        iceberg_configs.insert(S3_ENDPOINT.to_owned(), endpoint.clone());
                    }
                    if let Some(access_key) = &self.access_key {
                        iceberg_configs.insert(S3_ACCESS_KEY_ID.to_owned(), access_key.clone());
                    }
                    if let Some(secret_key) = &self.secret_key {
                        iceberg_configs.insert(S3_SECRET_ACCESS_KEY.to_owned(), secret_key.clone());
                    }
                };

                if let Some(credential) = &self.credential {
                    iceberg_configs.insert("credential".to_owned(), credential.clone());
                }
                if let Some(token) = &self.token {
                    iceberg_configs.insert("token".to_owned(), token.clone());
                }
                if let Some(oauth2_server_uri) = &self.oauth2_server_uri {
                    iceberg_configs
                        .insert("oauth2-server-uri".to_owned(), oauth2_server_uri.clone());
                }
                if let Some(scope) = &self.scope {
                    iceberg_configs.insert("scope".to_owned(), scope.clone());
                }

                let headers = self.headers()?;
                for (header_name, header_value) in headers {
                    iceberg_configs.insert(format!("header.{}", header_name), header_value);
                }

                let config_builder =
                    iceberg_catalog_rest::RestCatalogConfig::builder()
                        .uri(self.catalog_uri.clone().with_context(|| {
                            "`catalog.uri` must be set in rest catalog".to_owned()
                        })?)
                        .props(iceberg_configs);

                let config = match &self.warehouse_path {
                    Some(warehouse_path) => {
                        config_builder.warehouse(warehouse_path.clone()).build()
                    }
                    None => config_builder.build(),
                };
                let catalog = iceberg_catalog_rest::RestCatalog::new(config);
                Ok(Arc::new(catalog))
            }
            "glue_rust" => {
                let mut iceberg_configs = HashMap::new();
                // glue
                if let Some(region) = &self.region {
                    iceberg_configs.insert(AWS_REGION_NAME.to_owned(), region.clone());
                }
                if let Some(access_key) = &self.access_key {
                    iceberg_configs.insert(AWS_ACCESS_KEY_ID.to_owned(), access_key.clone());
                }
                if let Some(secret_key) = &self.secret_key {
                    iceberg_configs.insert(AWS_SECRET_ACCESS_KEY.to_owned(), secret_key.clone());
                }
                // s3
                if let Some(region) = &self.region {
                    iceberg_configs.insert(S3_REGION.to_owned(), region.clone());
                }
                if let Some(endpoint) = &self.endpoint {
                    iceberg_configs.insert(S3_ENDPOINT.to_owned(), endpoint.clone());
                }
                if let Some(access_key) = &self.access_key {
                    iceberg_configs.insert(S3_ACCESS_KEY_ID.to_owned(), access_key.clone());
                }
                if let Some(secret_key) = &self.secret_key {
                    iceberg_configs.insert(S3_SECRET_ACCESS_KEY.to_owned(), secret_key.clone());
                }
                let config_builder =
                    iceberg_catalog_glue::GlueCatalogConfig::builder()
                        .warehouse(self.warehouse_path.clone().ok_or_else(|| {
                            anyhow!("`warehouse.path` must be set in glue catalog")
                        })?)
                        .props(iceberg_configs);
                let config = if let Some(uri) = self.catalog_uri.as_deref() {
                    config_builder.uri(uri.to_owned()).build()
                } else {
                    config_builder.build()
                };
                let catalog = iceberg_catalog_glue::GlueCatalog::new(config).await?;
                Ok(Arc::new(catalog))
            }
            catalog_type
                if catalog_type == "hive"
                    || catalog_type == "snowflake"
                    || catalog_type == "jdbc"
                    || catalog_type == "rest"
                    || catalog_type == "glue" =>
            {
                // Create java catalog
                let (file_io_props, java_catalog_props) =
                    self.build_jni_catalog_configs(java_catalog_props)?;
                let catalog_impl = match catalog_type {
                    "hive" => "org.apache.iceberg.hive.HiveCatalog",
                    "jdbc" => "org.apache.iceberg.jdbc.JdbcCatalog",
                    "snowflake" => "org.apache.iceberg.snowflake.SnowflakeCatalog",
                    "rest" => "org.apache.iceberg.rest.RESTCatalog",
                    "glue" => "org.apache.iceberg.aws.glue.GlueCatalog",
                    _ => unreachable!(),
                };

                jni_catalog::JniCatalog::build_catalog(
                    file_io_props,
                    self.catalog_name(),
                    catalog_impl,
                    java_catalog_props,
                )
            }
            "mock" => Ok(Arc::new(mock_catalog::MockCatalog {})),
            _ => {
                bail!(
                    "Unsupported catalog type: {}, only support `storage`, `rest`, `hive`, `jdbc`, `glue`, `snowflake`",
                    self.catalog_type()
                )
            }
        }
    }

    /// TODO: remove the arguments and put them into `IcebergCommon`. Currently the handling in source and sink are different, so pass them separately to be safer.
    pub async fn load_table(
        &self,
        java_catalog_props: &HashMap<String, String>,
    ) -> ConnectorResult<Table> {
        let catalog = self
            .create_catalog(java_catalog_props)
            .await
            .context("Unable to load iceberg catalog")?;

        let table_id = self
            .full_table_name()
            .context("Unable to parse table name")?;

        catalog.load_table(&table_id).await.map_err(Into::into)
    }
}
