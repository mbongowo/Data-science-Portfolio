# Data Engineering Pipeline — weather lakehouse with dbt and Prefect

The Data Engineering Pipeline is an end-to-end weather data platform. It ingests
historical weather from the Open-Meteo API into a partitioned Parquet data lake,
loads it into DuckDB (and optionally a Terraform-provisioned Azure SQL database),
and transforms it with dbt models covered by 28 tests spanning generic, singular
and source-freshness checks. Prefect orchestrates the flow and a Streamlit
dashboard serves the results.

The project demonstrates a modern batch ELT stack: API ingestion, a Parquet lake,
a warehouse, dbt transformation with data-quality testing, orchestration and a
serving layer, all reproducible locally.
