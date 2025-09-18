"""
[description]
Airport distance calculator. Computes minimum distance from each feature to
airports for requested years using DuckDB Spatial over geometry chunks.
"""

from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path

VALID_YEARS = [2000, 2005, 2010, 2015, 2020]
TABLE_NAME = "airport"
VAR_NAME_MACRO = """
    CREATE OR REPLACE MACRO varname() AS (
        'D_Airport'
    );
"""


@typechecked
def _normalize_params(years: int | list[int]) -> list[int]:
    # normalize input type
    if isinstance(years, int):
        years = [years]
    # sort
    years = sorted(years)
    # check
    for year in years:
        if year not in VALID_YEARS:
            msg = f"Invalid year '{year}'. Valid years are: {VALID_YEARS}"
            raise ValueError(msg)
    # return
    return years

@typechecked
def _generate_query(year: int, table_path: Path) -> Tuple[str, str, str]:
    pre_query = VAR_NAME_MACRO
    main_query = f"""
        WITH 
        airport_sel_year AS (
            SELECT geometry
            FROM '{table_path}'
            WHERE year = {year} AND NOT ST_IsEmpty(geometry)
        ),
        result AS (
            SELECT 
                id,
                varname() AS varname,
                {year} AS year,
                MIN(ST_Distance(a.geometry, c.geometry)) AS value
            FROM chunk AS c, airport_sel_year AS a
            GROUP BY c.id
        )
        SELECT * FROM result;
    """
    post_query = ""
    return pre_query, main_query, post_query


class AirportDistanceCalculator:

    @typechecked
    def calculate_airport_distance(self, years: int | list[int]) -> Self:
        """
        [description]
        Calculate per-feature minimum distance to airports for one or more years.

        [input]
        - years: int | list[int] — Year(s) to compute (in `VALID_YEARS`).

        [output]
        - Self — Appends rows to `self.result_df` and returns self.

        [example usage]
        ```python
        calculator.calculate_airport_distance(years=[2000, 2005])
        ```
        """
        # normalize input
        years = _normalize_params(years)
        # generate requests per year
        table_path = (self.data_dir / f"{TABLE_NAME}.parquet")
        for year in years:
            pre_query, main_query, post_query = _generate_query(year, table_path)
            desc = f"Airport distance ({year})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self