"""
[description]
Road LLW calculator. Computes length-based road metrics within buffers per
feature and year: L (length), LL (lane length), LLW (lane length weighted by width).
"""
from typeguard import typechecked
from typing import Self, Tuple

VALID_MR_TYPES = ["mr1", "mr2"]
ROAD_TABLE_NAME = "roads"
VALID_YEARS = [2005, 2010, 2015, 2020]
VAR_NAME_MACRO = """
CREATE OR REPLACE MACRO varname(mr_type, vartype, buffer_size) AS (
    -- vartype: L (lane), LL (lane length), W (width)
    printf('%s_%s_%04d', 
        CASE
            WHEN mr_type = 'mr1' THEN 'MR1'
            WHEN mr_type = 'mr2' THEN 'MR2'
            ELSE 'error_processing_relative_elevation'
        END, 
        vartype, 
        buffer_size::Integer
    )
);
"""


@typechecked
def _normalize_params(mr_types: str | list[str], 
                      buffer_sizes: float | list[float], 
                      years: int | list[int], ) -> Tuple[list[str], list[int], list[float]]:
    """
    [description]
    Normalize and validate `mr_types`, `buffer_sizes`, and `years` arguments.

    [input]
    - mr_types: str | list[str] — One or more road types (in `VALID_MR_TYPES`).
    - buffer_sizes: float | list[float] — One or more buffer sizes.
    - years: int | list[int] — One or more target years (in `VALID_YEARS`).

    [output]
    - tuple[list[str], list[int], list[float]] — Sorted mr_types, years, and buffer sizes.
    """
    # normalize input type
    if isinstance(mr_types, str):
        mr_types = [mr_types]
    if isinstance(years, int):
        years = [years]
    if isinstance(buffer_sizes, float):
        buffer_sizes =[buffer_sizes]
    # sort
    mr_types = sorted(mr_types)
    years = sorted(years)
    buffer_sizes = sorted(buffer_sizes)
    # check
    for mr_type in mr_types:
        if mr_type not in VALID_MR_TYPES:
            msg = f"Invalid road type '{mr_type}'. Valid types are: {VALID_MR_TYPES}"
            raise ValueError(msg)
    for year in years:
        if year not in VALID_YEARS:
            msg = f"Invalid year '{year}'. Valid years are: {VALID_YEARS}"
            raise ValueError(msg)
    # return
    return mr_types, buffer_sizes, years


@typechecked
def _generate_query(mr_type: str, buffer_sizes: list[float], year: int, mr_table_path: str, road_table_path: str) -> Tuple[str, str, str]:
    """
    [description]
    Generate DuckDB SQL queries (pre, main, post) for calculating road metrics within buffers.
    
    [input]
    - mr_type: str — Road type ('mr1' or 'mr2').
    - buffer_sizes: list[float] — List of buffer sizes.
    - year: int — Target year for computation.
    - mr_table_path: str — Path to the main road table.
    - road_table_path: str — Path to the road table.
    
    [output]
    - Tuple[str, str, str] — Pre-query, main query, and post-query SQL strings.
    """
    values_clause = ", ".join(f"({bs})" for bs in buffer_sizes)
    pre_query = VAR_NAME_MACRO
    main_query = f"""
        WITH 
        buffer_sizes AS (
            SELECT * FROM (VALUES {values_clause}) AS t(buffer_size)
        ),
        mr_sel_year AS (
            SELECT roads_{year}_id AS id
            FROM '{mr_table_path}'
            WHERE roads_{year}_id IS NOT NULL
        ), 
        road_sel AS (
            SELECT r.lanes, r.width, r.geometry
            FROM '{road_table_path}' AS r
            INNER JOIN mr_sel_year AS mr ON r.id = mr.id
        ), 
        intersected AS (
            SELECT 
                c.id, 
                bs.buffer_size AS buffer_size,
                {year} AS year,
                r.lanes,
                r.width,
                ST_Intersection(ST_Buffer(c.geometry, bs.buffer_size), r.geometry) AS geometry
            FROM 
                chunk AS c
            CROSS JOIN 
                buffer_sizes AS bs
            LEFT JOIN 
                road_sel AS r ON ST_Intersects(ST_Buffer(c.geometry, bs.buffer_size), r.geometry)
        ),
        agg AS (
            SELECT 
                id, 
                buffer_size,
                year,
                COALESCE(SUM(ST_Length(geometry)), 0) AS L,
                COALESCE(SUM(ST_Length(geometry) * lanes), 0) AS LL,
                COALESCE(SUM(ST_Length(geometry) * lanes * width), 0) AS LLW
            FROM 
                intersected
            GROUP BY 
                id, buffer_size, year
        ), 
        unpivoted AS (
            UNPIVOT agg
            ON L, LL, LLW
            INTO NAME vartype VALUE value
        ), 
        result AS (
            SELECT
                id, 
                year,
                varname('{mr_type}', vartype, buffer_size) AS varname,
                value
            FROM unpivoted
        )
        SELECT * FROM result;
    """
    post_query = """
        drop macro varname;
    """
    return pre_query, main_query, post_query


class MainRoadLLWCalculator:
    """
    [description]
    Calculator for computing road metrics (L, LL, LLW) within buffers around features.
    Uses DuckDB Spatial to process geometry chunks in parallel or single-threaded mode.
    """

    @typechecked
    def calculate_main_road_llw(self, mr_types: str | list[str], buffer_sizes: float | list[float], years: int | list[int]) -> Self:
        """
        [description]
        Calculate road metrics (L, LL, LLW) within buffer(s) for one or more years and road types.

        [input]
        - mr_types: str | list[str] — Road type(s) ('mr1' or 'mr2').
        - buffer_sizes: float | list[float] — Buffer sizes to apply.
        - years: int | list[int] — Year(s) to compute (in `VALID_YEARS`).

        [output]
        - Self — Appends rows to `self.result_df` and returns self.

        [example usage]
        ```python
        calculator.calculate_main_road_llw(mr_types=['mr1'], buffer_sizes=[100, 300], years=[2010, 2020])
        ```
        """
        # normalize input
        mr_types, buffer_sizes, years = _normalize_params(mr_types, buffer_sizes, years)
        # generate request
        for mr_type in mr_types:
            mr_table_path = f"{self.data_dir}/{mr_type}.parquet"
            road_table_path = f"{self.data_dir}/{ROAD_TABLE_NAME}.parquet"
            for year in years:
                pre_query, main_query, post_query = _generate_query(
                    mr_type=mr_type, 
                    buffer_sizes=buffer_sizes, 
                    year=year, 
                    mr_table_path=mr_table_path, 
                    road_table_path=road_table_path
                )
                desc = f"{mr_type} LLW ({year}, {buffer_sizes})"
                self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self
