-- Staging: one clean, typed row per title. Renames IMDb's camelCase columns to
-- snake_case, casts numerics, and maps "\N" (already NULL after seed) through.
-- No business logic here; that lives in intermediate/marts.

with source as (

    select * from {{ source('raw', 'title_basics') }}

)

select
    tconst                              as title_id,
    titleType                           as title_type,
    primaryTitle                        as primary_title,
    originalTitle                       as original_title,
    cast(isAdult as integer)            as is_adult,
    try_cast(startYear as integer)      as start_year,
    try_cast(endYear as integer)        as end_year,
    try_cast(runtimeMinutes as integer) as runtime_minutes,
    genres                              as genres
from source
