"""
[description]
Bus stop distance calculator that computes per-feature minimum distances to bus stops
for requested years using DuckDB Spatial over geometry chunks.
"""

from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path

VALID_YEARS = [2020, 2021, 2023]
TABLE_NAME = "bus_stop"
VAR_NAME_MACRO = """
    CREATE OR REPLACE MACRO varname() AS (
        'D_Bus'
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
        busstop_sel_year AS (
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
            FROM chunk AS c, busstop_sel_year AS a
            GROUP BY c.id
        )
        SELECT * FROM result;
    """
    post_query = ""
    return pre_query, main_query, post_query


class BusStopDistanceCalculator:

    @typechecked
    def calculate_bus_stop_distance(self, years: int | list[int]) -> Self:
        """
        [description]
        Calculate per-feature minimum distance to airports for one or more years. Uses the
        standardized worker runner to execute DuckDB SQL per chunk and aggregates results
        into `self.result_df`.

        [input]
        - years: int | list[int] — One or more years to compute. Valid: `VALID_YEARS`.

        [output]
        - Self — Returns self for method chaining. Appends rows to `self.result_df` with columns
          [`id`, `varname`, `year`, `value`].

        [example usage]
        ```python
        geovariable = (
            calculator
            .set_dataframe(gdf)
            .chunk_by_centroid(max_cluster_size=MAX_CLUSTER_SIZE, distance_threshold=MAX_CLUSTER_WIDTH)
            .calculate_bus_stop_distance(years=[2020, 2021, 2023])
            .get_result(pivot=True)
        )
        """
        # normalize input
        years = _normalize_params(years)
        # generate requests per year
        table_path = (self.data_dir / f"{TABLE_NAME}.parquet")
        for year in years:
            pre_query, main_query, post_query = _generate_query(year, table_path)
            desc = f"Bus stop distance ({year})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self