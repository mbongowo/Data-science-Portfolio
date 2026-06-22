-- Mart: the rating fact. One row per rated title, carrying the foreign key to
-- dim_title (title_key) plus the measures. A simple rating band is derived for
-- the BI layer and covered by an accepted_values test.

with rated as (

    select * from {{ ref('int_titles_rated') }}

)

select
    {{ dbt_utils.generate_surrogate_key(['title_id']) }} as title_key,
    title_id,
    title_type,
    average_rating,
    num_votes,
    case
        when average_rating >= 8.0 then 'high'
        when average_rating >= 5.0 then 'medium'
        else 'low'
    end as rating_band
from rated
