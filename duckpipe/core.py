import geopandas as gpd
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
from duckpipe.duckdb_utils import generate_duckdb_connection, install_duckdb_extensions, generate_duckdb_memory_connection




class Calculator(Clustering,
                 CoordinateCalculator, 
                 LanduseCalculator, 
                 AirportDistanceCalculator, 
                 CoastlineDistanceCalculator, 
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

    @typechecked
    def set_dataframe(self, gdf: gpd.GeoDataFrame) -> Self:
        """
        [description]
        Register the input GeoDataFrame, transform to the reference CRS, convert geometries to WKT
        for DuckDB, sort by Hilbert distance for spatial locality, and pre-compute default chunks.

        [input]
        - gdf: geopandas.GeoDataFrame — Must contain a valid geometry column in EPSG:4326.

        [output]
        - Self — The same Calculator instance for method chaining.

        [example usage]
        ```python
        calculator = dp.Calculator(db_path="path/to/parquet_dir")
        calculator = calculator.set_dataframe(gdf)
        ```
        """
        # init.
        self.gdf = gdf
        self.geom_col = self.gdf.geometry.name
        # prepare
        self.gdf[C.ID_COL] = [str(i) for i in range(len(self.gdf))]
        self.geom_df = self.gdf.loc[:, [C.ID_COL, self.geom_col]]
        self.geom_df[self.geom_col] = self.geom_df[self.geom_col].to_crs(C.REF_EPSG)
        # wkt, for duckdb compatibility
        self.geom_df = (
            self.geom_df
            .assign(wkt=lambda df: df[self.geom_col].apply(lambda g: g.wkt))
            .drop(columns=[self.geom_col])
        )
        # chunking
        self.chunk_by_centroid(max_cluster_size=100, distance_threshold=10000)
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
        result_df = self.result_df.copy()
        result_df.sort_values(by=[C.ID_COL, C.YEAR_COL, C.VAR_COL], inplace=True)
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
        result = (
            self.gdf
            .drop(columns=[self.geom_col])
            .merge(result_df, on=C.ID_COL, how="left")
            .sort_values(by=[C.ID_COL, C.YEAR_COL])
            .drop(columns=[C.ID_COL])
            .reset_index(drop=True)
        )
        if self.verbose:
            print(f"Elapsed time: {self.end_time - self.start_time}")
        return result
