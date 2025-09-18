"""
Main road distance calculator that computes per-feature minimum distances to main roads
for requested types and years using DuckDB Spatial over geometry chunks.
"""
from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path

VALID_YEARS = [2005, 2010, 2015, 2020]
TABLE_NAME = "roads"
VAR_NAME_MACRO = """
    CREATE OR REPLACE MACRO varname() AS (
        'D_Road'
    );
"""


@typechecked
def _normalize_params(years: int | list[int]) -> list[int]:
    """
    Normalize and validate `mr_types` and `years` arguments.

    - years: int | list[int] — One or more target years.

    - list[int] — Sorted, validated years.
    """
    # normalize input type
    if isinstance(years, int):
        years =[years]
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
    """
    Build DuckDB SQL segments to compute minimum distance from each feature to the
    specified main road type for a given `year`.

    - year: int — Target year (one of `VALID_YEARS`).
    - table_path: pathlib.Path — Parquet table path for the given road type.

    - tuple[str, str, str] — (pre_query, main_query, post_query).
    """
    pre_query = VAR_NAME_MACRO
    main_query = f"""
        WITH 
        road_sel_year AS (
            SELECT geometry
            FROM '{table_path}'
            WHERE year = {year}
        ), 
        result AS (
            SELECT 
                id, 
                varname() AS varname, 
                {year} AS year, 
                MIN(ST_Distance(r.geometry, c.geometry)) AS value
            FROM chunk AS c, road_sel_year AS r
            GROUP BY c.id
        )
        SELECT * FROM result;
    """
    post_query = ""
    return pre_query, main_query, post_query


class RoadDistanceCalculator:

    def calculate_road_distance(self, years: int | list[int]) -> Self:
        """
        Calculate per-feature minimum distance to main roads for one or more `years` using the standardized worker runner (`run_query_workers`).

        - years: int | list[int] — Year(s) to compute (must be in `VALID_YEARS`).

        - Self — Returns self for chaining. Appends rows with [`id`, `varname`, `year`, `value`].

        ```python
        calculator.calculate_road_distance(years=[2010, 2020])
        ```
        """
        # normalize input
        years = _normalize_params(years)
        # generate request
        for year in years:
            table_path = self.data_dir / f"{TABLE_NAME}.parquet"
            pre_query, main_query, post_query = _generate_query(year, table_path)
            desc = f"Road distance ({year})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self
