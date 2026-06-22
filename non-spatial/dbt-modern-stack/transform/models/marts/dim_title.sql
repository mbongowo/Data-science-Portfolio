-- Mart: the title dimension. One row per title, with a warehouse-owned
-- surrogate key (a stable hash of the natural key) alongside the natural key.
-- This mirrors dwh.dimensional.surrogate_key in SQL.

with titles as (

    select * from {{ ref('stg_titles') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['title_id']) }} as title_key,
    title_id,
    title_type,
    primary_title,
    original_title,
    start_year,
    runtime_minutes,
    genres
from titles
