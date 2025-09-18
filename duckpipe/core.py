import pandas as pd
from typeguard import typechecked
from typing import Self
from datetime import datetime
from pathlib import Path

import duckpipe.common as C
from duckpipe.calculator import discover_calculator_classes
from duckpipe.duckdb_utils import install_duckdb_extensions, generate_duckdb_memory_connection
from duckpipe.calculator.Worker import WorkerMode

UUID = "_35ab93c72f484478a4cab4233aa3d434"

class _AutoMixinMeta(type):
    """Metaclass that appends discovered calculator classes (Worker, Clustering,
    and all *Calculator mixins) to class bases, keeping their intended order.
    """
    def __new__(mcls, name, bases, namespace):
        # calculator classes
        discovered = discover_calculator_classes()
        # merge with existing bases
        ordered: list[type] = []
        for cls in (*discovered, *bases):
            if cls not in ordered:
                ordered.append(cls)
        new_bases = tuple(ordered)
        return super().__new__(mcls, name, new_bases, namespace)

class Calculator(metaclass=_AutoMixinMeta):
    """
    [description]
    High-level orchestrator that composes calculator mixins and runs geospatial
    computations on DuckDB Spatial with chunked processing and result aggregation.

    [example usage]
    ```python
    import duckpipe as dp

    calculator = dp.Calculator(data_dir="path/to/parquet_dir", n_workers=2)
    result = (
        calculator
        .add_point_with_table(df, x_col="lon", y_col="lat", epsg=4326)
        .chunk_by_centroid(max_cluster_size=100, distance_threshold=10000)
        .calculate_airport_distance(years=[2000, 2005])
        .calculate_landuse_area_ratio(years=[2000, 2005], buffer_sizes=[100, 300, 500])
        .calculate_relative_elevation(elev_types=["dem", "dsm"], buffer_sizes=[1000, 5000])
        .calculate_river_distance(years=[2023])
        .get_result(pivot=True)
    )
    ```
    """
    @typechecked
    def __init__(self, 
                 data_dir: str | Path, 
                 mode: WorkerMode | str=WorkerMode.CHUNKED_MULTI, 
                 n_workers: int=8, 
                 verbose: bool=True):
        """
        [description]
        Configure runtime and open an in-memory DuckDB connection with Spatial.

        [input]
        - data_dir: str | Path — Directory containing Parquet sources used by calculators
        - mode: WorkerMode | str — Execution mode (chunked multi/single or total single).
        - n_workers: int — Degree of parallelism or DuckDB threads depending on mode.
        - verbose: bool — Enable progress bars and timing logs.

        [output]
        - None — Side effects: installs extensions, creates connection, stores config.

        [example usage]
        ```python
        calculator = Calculator(data_dir="/data/geo", n_workers=4, verbose=True)
        ```
        """
        self.worker_mode = mode
        self.n_workers = n_workers
        self.verbose = verbose
        self.data_dir = Path(data_dir)
        install_duckdb_extensions()
        self.conn = generate_duckdb_memory_connection()
        self.start_time = datetime.now()

    @typechecked
    def add_point_with_table(self, 
                            df: pd.DataFrame, 
                            x_col: str = 'longitude', 
                            y_col: str = 'latitude', 
                            epsg: int = 4326) -> Self:
        """
        [description]
        Register a plain DataFrame with x/y columns, transform to reference CRS,
        and prepare geometry/attribute tables and initial chunks.

        [input]
        - df: pandas.DataFrame — Source table with coordinate columns.
        - x_col: str — X/longitude column name in `df`.
        - y_col: str — Y/latitude column name in `df`.
        - epsg: int — EPSG code of input coordinates (default: 4326).

        [output]
        - Self — Populates `self.wkt_df`, `self.attr_df`, initializes `self.chunks` and `self.result_df`.
        """
        # register input df
        self.conn.register('input_df', df)
        # geometry df
        query = f"""
        SELECT
            ROW_NUMBER() OVER () AS id, 
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
        SELECT *, ROW_NUMBER() OVER () AS {UUID}
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
        Assemble final output by merging computed variables back to attributes.
        Optionally pivot long-form rows to a wide table per (id, year).

        [input]
        - pivot: bool — Pivot to wide format if True; otherwise return long-form.

        [output]
        - pandas.DataFrame — Result with original columns plus computed variables.

        [example usage]
        ```python
        result = calculator.get_result(pivot=True)
        ```
        """
        self.end_time = datetime.now()
        result_df = self.result_df
        if pivot:
            result_df = result_df.pivot_table(
                index=["id", "year"], 
                columns=["varname"], 
                values="value",
                fill_value=None, 
                aggfunc="first", 
                dropna=False
            )
            result_df = result_df[sorted(result_df.columns)]
            result_df.reset_index(inplace=True)
        else:
            result_df.sort_values(by=["id", "year", "varname"], inplace=True)
        
        # merge
        result = (
            self.attr_df
            .merge(result_df, left_on=UUID, right_on="id", how="left", suffixes=("", "_"))
            .sort_values(by=[UUID, "year"])
            .drop(columns=[UUID])
            .reset_index(drop=True)
        )
        if "id_" in result.columns:
            result.drop(columns=["id_"], inplace=True)
        if self.verbose:
            print(f"Elapsed time: {self.end_time - self.start_time}")
        return result
