-- Staging: one clean, typed row per rated title.

with source as (

    select * from {{ source('raw', 'title_ratings') }}

)

select
    tconst                         as title_id,
    cast(averageRating as double)  as average_rating,
    cast(numVotes as integer)      as num_votes
from source
