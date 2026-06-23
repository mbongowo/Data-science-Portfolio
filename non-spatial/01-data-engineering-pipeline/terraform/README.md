# Infrastructure as Code

Two paths, only one of which touches a cloud:

## Free path (default) — no cloud

The local path uses **DuckDB**, a single file with zero infrastructure. It needs
**no Terraform and no Azure account**. Run the pipeline end to end for free:

```bash
python -m weatherpipe.cli demo        # core on synthetic data, no warehouse
# or the real flow:
python -m weatherpipe.cli ingest      # Open-Meteo -> data/lake (Parquet)
python -m weatherpipe.cli build       # lake -> data/warehouse.duckdb + dbt build
```

## Azure path (opt-in) — costs money

`terraform/azure/` provisions a cloud lake (ADLS Gen2) and warehouse (Azure SQL
serverless) so the **same dbt models** run against Azure SQL via the `azure`
profile.

> **This path costs money.** `terraform validate` / `terraform plan` are free and
> change nothing. **Never run `terraform apply` without explicitly deciding to
> incur the cost**, and run `terraform destroy` when you are done. See
> [`azure/README.md`](azure/README.md) for the full steps, the cost warning, and
> the teardown note.
