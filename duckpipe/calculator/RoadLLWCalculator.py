"""
[description]
Road LLW calculator. Computes length-based road metrics within buffers per
feature and year: L (length), LL (lane length), LLW (lane length weighted by width).
"""
from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path

TABLE_NAME = "roads"
VALID_YEARS = [2005, 2010, 2015, 2020]
VAR_NAME_MACRO = """
CREATE OR REPLACE MACRO varname(vartype, buffer_size) AS (
    -- vartype: L (lane), LL (lane length), W (width)
    printf('%s_%s_%04d', 'Road', vartype, buffer_size::INTEGER)
);
"""


@typechecked
def _normalize_params(buffer_sizes: float | list[float], years: int | list[int], ) -> Tuple[list[int], list[float]]:
    """
    [description]
    Normalize and validate `buffer_sizes` and `years` arguments.

    [input]
    - buffer_sizes: float | list[float] — One or more buffer sizes.
    - years: int | list[int] — One or more target years (in `VALID_YEARS`).

    [output]
    - tuple[list[float], list[int]] — Sorted buffer sizes and years.
    """
    # normalize input type
    if isinstance(years, int):
        years = [years]
    if isinstance(buffer_sizes, float):
        buffer_sizes =[buffer_sizes]
    # sort
    years = sorted(years)
    buffer_sizes = sorted(buffer_sizes)
    # check
    for year in years:
        if year not in VALID_YEARS:
            msg = f"Invalid year '{year}'. Valid years are: {VALID_YEARS}"
            raise ValueError(msg)
    # return
    return buffer_sizes, years


@typechecked
def _generate_query(buffer_sizes: list[float], year: int, table_path: str) -> Tuple[str, str, str]:
    """
    [description]
    Build DuckDB SQL to compute road metrics (L, LL, LLW) within given buffers for `year`.

    [input]
    - buffer_sizes: list[float] — Buffer sizes to apply.
    - year: int — Target year (in `VALID_YEARS`).
    - table_path: str — Parquet table path for roads.

    [output]
    - tuple[str, str, str] — (pre_query, main_query, post_query).
    """
    values_clause = ", ".join(f"({bs})" for bs in buffer_sizes)
    pre_query = VAR_NAME_MACRO
    main_query = f"""
        WITH 
        buffer_sizes AS (
            SELECT * FROM (VALUES {values_clause}) AS t(buffer_size)
        ),
        road_sel_year AS (
            SELECT geometry, lanes, width, year
            FROM '{table_path}'
            WHERE year = {year}
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
            CROSS JOIN buffer_sizes AS bs
            LEFT JOIN road_sel_year AS r 
                ON ST_Intersects(ST_Buffer(c.geometry, bs.buffer_size), r.geometry)
        ),
        agg AS (
            SELECT 
                id, 
                buffer_size,
                year,
                COALESCE(SUM(ST_Length(geometry)), 0) AS L,
                COALESCE(SUM(ST_Length(geometry) * lanes), 0) AS LL,
                COALESCE(SUM(ST_Length(geometry) * lanes * width), 0) AS LLW
            FROM intersected
            GROUP BY id, buffer_size, year
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
                varname(vartype, buffer_size) AS varname,
                value
            FROM unpivoted
        )
        SELECT * FROM result;
    """
    post_query = """
        drop macro varname;
    """
    return pre_query, main_query, post_query


class RoadLLWCalculator:

    def calculate_road_llw(self, buffer_sizes: float | list[float], years: int | list[int]) -> Self:
        """
        [description]
        Calculate road metrics (L, LL, LLW) within buffer(s) for one or more years.

        [input]
        - buffer_sizes: float | list[float] — Buffer sizes to apply.
        - years: int | list[int] — Year(s) to compute (in `VALID_YEARS`).

        [output]
        - Self — Appends rows to `self.result_df` and returns self.

        [example usage]
        ```python
        calculator.calculate_road_llw(buffer_sizes=[100, 300], years=[2010, 2020])
        ```
        """
        # normalize input
        buffer_sizes, years = _normalize_params(buffer_sizes, years)
        # generate request
        for year in years:
            table_path = f"{self.data_dir}/{TABLE_NAME}.parquet"
            pre_query, main_query, post_query = _generate_query(buffer_sizes, year, table_path)
            desc = f"Road LLW ({year}, {buffer_sizes})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self
