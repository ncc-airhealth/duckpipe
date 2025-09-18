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
    Normalize and validate `mr_types` and `years` arguments.

    - var_types: str | list[str] — One or more variable types (e.g., "L", "LL", "W").
    - buffer_sizes: float | list[float] — One or more buffer sizes.

    - (list[str], list[float]) — Sorted, validated variable types and buffer sizes.
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
def _generate_query(buffer_sizes: list[float], year: int, table_path: Path) -> Tuple[str, str, str]:
    """
    Build DuckDB SQL segments to compute minimum distance from each feature to the
    specified main road type for a given `year`.

    - mr_type: str — Road type key (one of `VALID_MR_TYPES`).
    - year: int — Target year (one of `VALID_YEARS`).
    - table_path: pathlib.Path — Parquet table path for the given road type.

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
        Calculate per-feature minimum distance to main roads for one or more `mr_types`
        and `years` using the standardized worker runner (`run_query_workers`).

        - mr_types: str | list[str] — Road type(s) to compute (must be in `VALID_MR_TYPES`).
        - years: int | list[int] — Year(s) to compute (must be in `VALID_YEARS`).

        - Self — Returns self for chaining. Appends rows with [`id`, `varname`, `year`, `value`].

        ```python
        calculator.calculate_road_llw(mr_types=["mr1", "mr2"], years=[2010, 2020])
        ```
        """
        # normalize input
        buffer_sizes, years = _normalize_params(buffer_sizes, years)
        # generate request
        for year in years:
            table_path = self.data_dir / f"{TABLE_NAME}.parquet"
            pre_query, main_query, post_query = _generate_query(buffer_sizes, year, table_path)
            desc = f"Road LLW ({year}, {buffer_sizes})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self
