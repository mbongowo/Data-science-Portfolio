-- Staging: one clean, typed row per station/day. Casts the measures to double,
-- the date to a real date, and trims the station label. No business logic here;
-- that lives in the marts. Rows with a null key or measure are filtered so the
-- downstream marts get a clean daily grain (the pure-pandas validator does the
-- same job before load).

with source as (

    select * from {{ source('raw', 'weather') }}

)

select
    trim(station)              as station,
    cast(date as date)         as date,
    cast(tmin_c as double)     as tmin_c,
    cast(tmax_c as double)     as tmax_c,
    cast(tmean_c as double)    as tmean_c,
    cast(precip_mm as double)  as precip_mm
from source
where station is not null
  and date is not null
  and tmin_c is not null
  and tmax_c is not null
  and tmean_c is not null
  and precip_mm is not null
