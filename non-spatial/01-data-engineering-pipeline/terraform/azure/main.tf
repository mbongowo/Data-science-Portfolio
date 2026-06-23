# Azure infrastructure for the opt-in cloud path of the Cameroon weather pipeline.
#
# WARNING: applying this provisions billable Azure resources (a Storage Account
# with ADLS Gen2 and an Azure SQL Database). It COSTS MONEY. `terraform validate`
# and `terraform plan` are free and safe; never `terraform apply` without an
# explicit decision, and run `terraform destroy` when you are done. See the
# README in this directory.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.110"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  subscription_id = var.subscription_id
  features {}
}

# A short random suffix keeps the globally-unique storage-account name available.
resource "random_string" "suffix" {
  length  = 6
  upper   = false
  special = false
}

# ---------------------------------------------------------------------------- #
# Resource group: the container for everything below.
# ---------------------------------------------------------------------------- #
resource "azurerm_resource_group" "weather" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# ---------------------------------------------------------------------------- #
# Lake: a Storage Account with ADLS Gen2 (hierarchical namespace) + a `raw`
# container. This is the cloud equivalent of the local data/lake directory.
# ---------------------------------------------------------------------------- #
resource "azurerm_storage_account" "lake" {
  name                     = "${var.storage_account_prefix}${random_string.suffix.result}"
  resource_group_name      = azurerm_resource_group.weather.name
  location                 = azurerm_resource_group.weather.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  account_kind             = "StorageV2"
  # Hierarchical namespace = ADLS Gen2 (directory semantics for the lake).
  is_hns_enabled           = true
  min_tls_version          = "TLS1_2"
  tags                     = var.tags
}

resource "azurerm_storage_data_lake_gen2_filesystem" "raw" {
  name               = "raw"
  storage_account_id = azurerm_storage_account.lake.id
}

# ---------------------------------------------------------------------------- #
# Warehouse: an Azure SQL Database on the serverless tier (auto-pauses when idle
# to keep the bill low). dbt targets this via the `azure` profile.
# ---------------------------------------------------------------------------- #
resource "azurerm_mssql_server" "warehouse" {
  name                         = "${var.sql_server_prefix}-${random_string.suffix.result}"
  resource_group_name          = azurerm_resource_group.weather.name
  location                     = azurerm_resource_group.weather.location
  version                      = "12.0"
  administrator_login          = var.sql_admin_login
  administrator_login_password = var.sql_admin_password
  minimum_tls_version          = "1.2"
  tags                         = var.tags
}

resource "azurerm_mssql_database" "weather" {
  name      = var.sql_database_name
  server_id = azurerm_mssql_server.warehouse.id
  # Serverless General Purpose, smallest size; auto-pauses after 60 idle minutes.
  sku_name                    = "GP_S_Gen5_1"
  min_capacity                = 0.5
  auto_pause_delay_in_minutes = 60
  max_size_gb                 = 32
  collation                   = "SQL_Latin1_General_CP1_CI_AS"
  tags                        = var.tags
}

# Allow Azure services (and, optionally, your client IP) to reach the server.
# Tighten or remove this for anything beyond a personal sandbox.
resource "azurerm_mssql_firewall_rule" "allow_azure" {
  name             = "allow-azure-services"
  server_id        = azurerm_mssql_server.warehouse.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_mssql_firewall_rule" "allow_client" {
  count            = var.client_ip_address == "" ? 0 : 1
  name             = "allow-client-ip"
  server_id        = azurerm_mssql_server.warehouse.id
  start_ip_address = var.client_ip_address
  end_ip_address   = var.client_ip_address
}
