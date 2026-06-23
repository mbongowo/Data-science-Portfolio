-- Mart: per station/month aggregate read by the dashboard. The warehouse mirror
-- of weatherpipe.transform.monthly_summary: mean/min/max temperature, total
-- precipitation, the rain-day count and the record count for the month.

with daily as (

    select * from {{ ref('daily_weather') }}

)

select
    station,
    year,
    month,
    avg(tmean_c)                        as tmean_mean,
    min(tmin_c)                         as tmin_min,
    max(tmax_c)                         as tmax_max,
    sum(precip_mm)                      as precip_total_mm,
    sum(case when is_rain_day then 1 else 0 end) as rain_days,
    count(*)                           as record_count
from daily
group by station, year, month
