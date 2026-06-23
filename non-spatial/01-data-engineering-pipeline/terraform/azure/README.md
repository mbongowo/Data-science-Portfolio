# Azure deployment (opt-in, costs money)

This module provisions the cloud warehouse and lake for the Azure path of the
pipeline:

- a **resource group**,
- a **Storage Account** with **ADLS Gen2** (hierarchical namespace) and a `raw`
  container — the cloud lake,
- an **Azure SQL Database** on the **serverless** tier — the warehouse that dbt
  targets via the `azure` profile.

## Cost warning

> **Applying this creates billable Azure resources.** The serverless SQL
> Database auto-pauses when idle, but storage, any active compute, and egress
> still cost money. `terraform validate` and `terraform plan` are free and make
> no changes. **Do not run `terraform apply` unless you have decided to incur the
> cost**, and run `terraform destroy` as soon as you are done. The free local
> path (DuckDB) needs none of this.

## Prerequisites

- Terraform >= 1.5
- Azure CLI logged in (`az login`) with a subscription you can deploy into
- A strong SQL admin password supplied via `TF_VAR_sql_admin_password`

## Validate and plan (free, safe)

```bash
cd terraform/azure
cp terraform.tfvars.example terraform.tfvars   # then edit subscription_id etc.
export TF_VAR_sql_admin_password='<a-strong-password>'   # PowerShell: $env:TF_VAR_...

terraform init
terraform fmt -check
terraform validate
terraform plan
```

`terraform plan` prints exactly what would be created without creating anything.

## Apply (opt-in — incurs cost)

Only if you have decided to incur the cost:

```bash
terraform apply    # review the plan, type "yes" to confirm
```

After apply, wire dbt to the warehouse by exporting the outputs into the
environment the `azure` dbt profile reads (see `.env.example` and
`transform/profiles.example.yml`):

```bash
export AZURE_SQL_SERVER="$(terraform output -raw sql_server_fqdn)"
export AZURE_SQL_DB="$(terraform output -raw sql_database_name)"
# AZURE_SQL_USER / AZURE_SQL_PASSWORD = the admin login / password you set
cd ../../transform && dbt build --target azure
```

## Teardown

```bash
terraform destroy   # removes every resource this module created
```

Destroying stops all charges from these resources. Confirm in the Azure portal
that the resource group is gone.
