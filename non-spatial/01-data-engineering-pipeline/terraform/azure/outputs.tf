output "resource_group_name" {
  description = "The resource group holding the pipeline resources."
  value       = azurerm_resource_group.weather.name
}

output "storage_account_name" {
  description = "The ADLS Gen2 storage account that backs the lake."
  value       = azurerm_storage_account.lake.name
}

output "lake_filesystem" {
  description = "The ADLS Gen2 filesystem (container) used for raw weather."
  value       = azurerm_storage_data_lake_gen2_filesystem.raw.name
}

output "sql_server_fqdn" {
  description = "Fully-qualified name of the Azure SQL server (use as AZURE_SQL_SERVER)."
  value       = azurerm_mssql_server.warehouse.fully_qualified_domain_name
}

output "sql_database_name" {
  description = "The Azure SQL Database used as the warehouse (use as AZURE_SQL_DB)."
  value       = azurerm_mssql_database.weather.name
}
