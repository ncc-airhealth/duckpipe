"""
[description]
Coastline distance calculator that computes per-feature minimum distances to the coastline
for requested years using DuckDB Spatial over geometry chunks.
"""

from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path


VALID_YEARS = [2000, 2005, 2010, 2015, 2020]
TABLE_NAME = "coastline"
VAR_NAME_MACRO = """
CREATE OR REPLACE MACRO varname() AS
    'D_Coast'
"""
SIMPLIFY_THRESHOLD = 1

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

def _generate_query(year: int, table_path: Path) -> Tuple[str, str, str]:
    pre_query = VAR_NAME_MACRO
    main_query = f"""
        WITH 
        coastline_sel_year AS (
            SELECT ST_Simplify(geometry, {SIMPLIFY_THRESHOLD}) AS geometry
            FROM '{table_path}'
            WHERE year = {year}
        )
        , result AS (
            SELECT 
                id,
                varname() AS varname,
                {year} AS year,
                MIN(ST_Distance(t.geometry, c.geometry)) AS value
            FROM chunk AS c, coastline_sel_year AS t
            GROUP BY c.id
        )
        SELECT * FROM result;
    """
    post_query = ""
    return pre_query, main_query, post_query


class CoastlineDistanceCalculator:

    @typechecked
    def calculate_coastline_distance(self, years: int | list[int]) -> Self:
        """
        [description]
        Calculate per-feature minimum distance to the coastline for one or more years using
        the standardized worker runner (`run_query_workers`).

        [input]
        - years: int | list[int] — One or more target years (must be in `VALID_YEARS`).

        [output]
        - Self — Returns self for chaining. Appends rows with [`id`, `varname`, `year`, `value`].
        """
        # normalize input
        years = _normalize_params(years)
        # run per-year
        table_path = (self.data_dir / TABLE_NAME).with_suffix(".parquet")
        for year in years:
            pre_query, main_query, post_query = _generate_query(year, table_path)
            desc = f"Coastline distance ({year})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self
