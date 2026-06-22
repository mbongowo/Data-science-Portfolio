# Data: NYC TLC trip records

This project runs on the **NYC Taxi & Limousine Commission (TLC) trip-record**
dataset. The TLC publishes one Parquet file per service per month, going back
years; the yellow-taxi series alone runs to **billions of rows**. That scale is
the whole point — it is what makes the engine bake-off meaningful.

No geospatial columns are used. Pickup/dropoff latitude/longitude (and, in newer
schemas, the location-zone IDs) are dropped on ingest. The analysis is purely
tabular: fares, tips, trip duration, payment type, and demand by hour and day of
week.

## Where to get it

Official page (Parquet downloads, per month, per service):

- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Each link is a direct Parquet URL of the form:

```
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_<YYYY>-<MM>.parquet
```

## How to lay it out

The engines expect a Hive-partitioned lake keyed on `year` and `month` (see
`config/tlc.yaml`). Download the months you want and arrange them as:

```
data/raw/yellow/
  year=2023/
    month=01/yellow_tripdata_2023-01.parquet
    month=02/yellow_tripdata_2023-02.parquet
    ...
```

A minimal download loop (bash) for one year:

```bash
YEAR=2023
for M in 01 02 03 04 05 06 07 08 09 10 11 12; do
  mkdir -p "data/raw/yellow/year=${YEAR}/month=${M}"
  curl -L -o "data/raw/yellow/year=${YEAR}/month=${M}/yellow_tripdata_${YEAR}-${M}.parquet" \
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${YEAR}-${M}.parquet"
done
```

Pull as many years as you need; `config/tlc.yaml` lists the years the workload
covers by default (2019-2023).

## Schema notes

Column names drift across years (e.g. `tpep_pickup_datetime`,
`payment_type`, `fare_amount`, `tip_amount`, `passenger_count`). The cleaning
layer (`src/tlc/clean.py`) works on a normalised tabular frame; map the raw
columns to that frame when you ingest a given year's schema.

Everything under `data/raw/` is git-ignored and reproducible from the links
above.
