import pandas as pd
from typeguard import typechecked
from typing import Self
from datetime import datetime
from pathlib import Path

import duckpipe.common as C
from duckpipe.calculator.Clustering import Clustering
from duckpipe.calculator.CoordinateCalculator import CoordinateCalculator
from duckpipe.calculator.LanduseCalculator import LanduseCalculator
from duckpipe.calculator.AirportDistanceCalculator import AirportDistanceCalculator
from duckpipe.calculator.CoastlineDistanceCalculator import CoastlineDistanceCalculator
from duckpipe.calculator.RelativeElevationCalculator import RelativeElevationCalculator
from duckpipe.calculator.MainRoadDistanceCalculator import MainRoadDistanceCalculator
from duckpipe.duckdb_utils import install_duckdb_extensions, generate_duckdb_memory_connection




class Calculator(Clustering,
                 CoordinateCalculator, 
                 LanduseCalculator, 
                 AirportDistanceCalculator, 
                 CoastlineDistanceCalculator, 
                 MainRoadDistanceCalculator,
                 RelativeElevationCalculator, 
                 ):
    """
    [description]
    High-level orchestrator that composes all calculator mixins to compute geospatial variables
    using DuckDB + Spatial. It manages DB initialization, chunking of input geometries, and
    aggregation of results into a final DataFrame.

    [example usage]
    ```python
    import duckpipe as dp

    calculator = dp.Calculator(db_path="path/to/parquet_dir", n_workers=2, memory_limit="6GB")
    result = (
        calculator
        .set_dataframe(gdf)  # a GeoDataFrame with geometry in EPSG:4326
        .chunk_by_centroid(max_cluster_size=100, distance_threshold=10000)
        .calculate_airport_distance(years=[2000, 2005])
        .calculate_coastline_distance(years=[2000, 2005])
        .calculate_landuse_area_ratio(years=[2000, 2005], buffer_sizes=[100, 300, 500])
        .calculate_relative_elevation(elevation_types=["dem", "dsm"], buffer_sizes=[1000, 5000])
        .get_result(pivot=True)
    )
    ```
    """
    @typechecked
    def __init__(self, db_path: str | Path, n_workers: int=8, memory_limit: str="5GB", verbose=True):
        """
        [description]
        Initialize the Calculator with an in-memory DuckDB connection and runtime configuration.

        [input]
        - db_path: str | Path — Directory path containing Parquet files used by calculators (e.g.,
          "airport.parquet", "coastline.parquet", "landuse_YYYY.parquet", "dem.parquet", "dsm.parquet").
        - n_workers: int — Number of worker processes used by calculator methods.
        - memory_limit: str — Memory limit passed to DuckDB (e.g., "6GB").
        - verbose: bool — If True, prints progress bars and elapsed time.

        [output]
        - None — Side effects: installs extensions, opens a connection, and stores configuration.

        [example usage]
        ```python
        import duckpipe as dp
        calculator = dp.Calculator(db_path="path/to/parquet_dir", n_workers=2, memory_limit="6GB")
        ```
        """
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.verbose = verbose
        self.db_path = Path(db_path)
        install_duckdb_extensions()
        self.conn = generate_duckdb_memory_connection(memory_limit=memory_limit)
        self.start_time = datetime.now()
        self.wkt_df = pd.DataFrame()
        self.attr_df = pd.DataFrame()

    @typechecked
    def add_point_with_table(self, 
                            df: pd.DataFrame, 
                            x_col: str = 'longitude', 
                            y_col: str = 'latitude', 
                            epsg: int = 4326) -> Self:
        # register input df
        self.conn.register('input_df', df)
        # geometry df
        query = f"""
        SELECT
            ROW_NUMBER() OVER () AS {C.ID_COL}, 
            ST_AsText(
                ST_Transform(
                    ST_Point({x_col}, {y_col}), 
                    'EPSG:{epsg}', 
                    'EPSG:{C.REF_EPSG}', 
                    always_xy := true
                )
            ) AS wkt
        FROM input_df
        """
        self.wkt_df = self.conn.execute(query).df()
        # attribute df
        query = f"""
        SELECT 
            ROW_NUMBER() OVER () AS {C.ID_COL}, 
            *
        FROM input_df
        """
        self.attr_df = self.conn.execute(query).df()
        # clean
        self.conn.unregister('input_df')
        # chunking
        self.chunk_by_order(max_cluster_size=100)
        # result preparation
        self.result_df = pd.DataFrame()
        return self
        
    @typechecked
    def get_result(self, pivot: bool=True) -> pd.DataFrame:
        """
        [description]
        Merge and shape the accumulated results into the final DataFrame. Optionally pivot the
        long-form results into a wide format with one row per (id, year).

        [input]
        - pivot: bool — If True, pivot to wide format; otherwise, return long-form results.

        [output]
        - pandas.DataFrame — Final result joined back to the original (non-geometry) columns.

        [example usage]
        ```python
        result = calculator.get_result(pivot=True)
        ```
        """
        self.end_time = datetime.now()
        result_df = self.result_df
        if pivot:
            result_df = result_df.pivot_table(
                index=[C.ID_COL, C.YEAR_COL], 
                columns=[C.VAR_COL], 
                values=C.VAL_COL,
                fill_value=None, 
                aggfunc="first", 
                dropna=False
            )
            result_df = result_df[sorted(result_df.columns)]
            result_df.reset_index(inplace=True)
        else:
            result_df.sort_values(by=[C.ID_COL, C.YEAR_COL, C.VAR_COL], inplace=True)
        result = (
            self.attr_df
            .merge(result_df, on=C.ID_COL, how="left")
            .sort_values(by=[C.ID_COL, C.YEAR_COL])
            .drop(columns=[C.ID_COL])
            .reset_index(drop=True)
        )
        if self.verbose:
            print(f"Elapsed time: {self.end_time - self.start_time}")
        return result
