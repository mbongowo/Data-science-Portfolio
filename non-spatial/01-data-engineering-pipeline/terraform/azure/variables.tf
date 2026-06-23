variable "subscription_id" {
  description = "Azure subscription ID to deploy into."
  type        = string
}

variable "resource_group_name" {
  description = "Name of the resource group that holds the pipeline resources."
  type        = string
  default     = "rg-cameroon-weather"
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "westeurope"
}

variable "storage_account_prefix" {
  description = "Prefix for the (globally-unique) ADLS Gen2 storage account name. A random suffix is appended. Lowercase letters and digits only, <= 18 chars."
  type        = string
  default     = "stcmrweather"

  validation {
    condition     = can(regex("^[a-z0-9]{3,18}$", var.storage_account_prefix))
    error_message = "storage_account_prefix must be 3-18 lowercase letters/digits."
  }
}

variable "sql_server_prefix" {
  description = "Prefix for the Azure SQL logical server name. A random suffix is appended."
  type        = string
  default     = "sql-cmr-weather"
}

variable "sql_database_name" {
  description = "Name of the Azure SQL Database used as the warehouse."
  type        = string
  default     = "weather_dwh"
}

variable "sql_admin_login" {
  description = "Administrator login for the Azure SQL server."
  type        = string
  default     = "weatheradmin"
}

variable "sql_admin_password" {
  description = "Administrator password for the Azure SQL server. Pass via TF_VAR_sql_admin_password or terraform.tfvars (never commit it)."
  type        = string
  sensitive   = true
}

variable "client_ip_address" {
  description = "Optional client IP to allow through the SQL firewall. Empty disables the rule."
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default = {
    project = "cameroon-weather-pipeline"
    owner   = "joseph-mbuh"
    env     = "sandbox"
  }
}
