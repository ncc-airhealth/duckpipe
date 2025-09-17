"""
Main road distance calculator that computes per-feature minimum distances to main roads
for requested types and years using DuckDB Spatial over geometry chunks.
"""
from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path

VALID_YEARS = [2005, 2010, 2015, 2020]
VALID_MR_TYPES = ["mr1", "mr2"]
VAR_NAME_MACRO = """
    CREATE OR REPLACE MACRO varname(mr_type) AS (
        CASE
            WHEN mr_type = 'mr1' THEN 'D_MR1'
            WHEN mr_type = 'mr2' THEN 'D_MR2'
            ELSE 'error_processing_main_road_distance'
        END
    );
"""


@typechecked
def _normalize_params(mr_types: str | list[str], years: int | list[int]) -> Tuple[list[str], list[int]]:
    """
    Normalize and validate `mr_types` and `years` arguments.

    - mr_types: str | list[str] — One or more road types (e.g., "mr1", "mr2").
    - years: int | list[int] — One or more target years.

    - (list[str], list[int]) — Sorted, validated road types and years.
    """
    # normalize input type
    if isinstance(mr_types, str):
        mr_types = [mr_types]
    if isinstance(years, int):
        years =[years]
    # sort
    mr_types = sorted(mr_types)
    years = sorted(years)
    # check
    for mr_type in mr_types:
        if mr_type not in VALID_MR_TYPES:
            msg = f"Invalid mr_type '{mr_type}'. Valid mr_types are: {VALID_MR_TYPES}"
            raise ValueError(msg)
    for year in years:
        if year not in VALID_YEARS:
            msg = f"Invalid year '{year}'. Valid years are: {VALID_YEARS}"
            raise ValueError(msg)
    # return
    return mr_types, years

@typechecked
def _generate_query(mr_type: str, year: int, table_path: Path) -> Tuple[str, str, str]:
    """
    Build DuckDB SQL segments to compute minimum distance from each feature to the
    specified main road type for a given `year`.

    - mr_type: str — Road type key (one of `VALID_MR_TYPES`).
    - year: int — Target year (one of `VALID_YEARS`).
    - table_path: pathlib.Path — Parquet table path for the given road type.

    - tuple[str, str, str] — (pre_query, main_query, post_query).
    """
    pre_query = VAR_NAME_MACRO
    main_query = f"""
        WITH 
        mr_sel_year AS (
            SELECT geometry
            FROM '{table_path}'
            WHERE year = {year}
        ), 
        result AS (
            SELECT 
                id, 
                varname('{mr_type}') AS varname, 
                {year} AS year, 
                MIN(ST_Distance(m.geometry, c.geometry)) AS value
            FROM chunk AS c, mr_sel_year AS m
            GROUP BY c.id
        )
        SELECT * FROM result;
    """
    post_query = ""
    return pre_query, main_query, post_query


class MainRoadDistanceCalculator:

    def calculate_main_road_distance(self, mr_types: str | list[str], years: int | list[int]) -> Self:
        """
        Calculate per-feature minimum distance to main roads for one or more `mr_types`
        and `years` using the standardized worker runner (`run_query_workers`).

        - mr_types: str | list[str] — Road type(s) to compute (must be in `VALID_MR_TYPES`).
        - years: int | list[int] — Year(s) to compute (must be in `VALID_YEARS`).

        - Self — Returns self for chaining. Appends rows with [`id`, `varname`, `year`, `value`].

        ```python
        calculator.calculate_main_road_distance(mr_types=["mr1", "mr2"], years=[2010, 2020])
        ```
        """
        # normalize input
        mr_types, years = _normalize_params(mr_types, years)
        # generate request
        for mr_type in mr_types:
            for year in years:
                table_path = self.data_dir / f"{mr_type}.parquet"
                pre_query, main_query, post_query = _generate_query(mr_type, year, table_path)
                desc = f"{mr_type} distance ({year})"
                self.run_query_workers(pre_query, main_query, post_query, desc=desc)
        # done
        return self
