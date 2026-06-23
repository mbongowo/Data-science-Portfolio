-- Mart: the clean daily grain. One row per station/day with a surrogate key and
-- a derived rain-day flag, the warehouse mirror of
-- weatherpipe.transform.daily_summary. The pure-pandas validator already drops
-- the bad rows; this is the curated daily table the monthly mart rolls up.

with daily as (

    select * from {{ ref('stg_weather') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['station', 'date']) }} as weather_key,
    station,
    date,
    extract(year from date)   as year,
    extract(month from date)  as month,
    tmin_c,
    tmax_c,
    tmean_c,
    precip_mm,
    case when precip_mm >= 1.0 then true else false end as is_rain_day
from daily
