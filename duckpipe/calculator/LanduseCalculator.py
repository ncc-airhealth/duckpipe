"""
Land-use calculator (Parquet backend).

This module computes land-use area and ratio statistics around input geometries
for requested `years` and `buffer_sizes`, using multiprocessing with an
in-memory DuckDB + Spatial connection. Land-use data is read from yearly
Parquet files named like "landuse_{year}.parquet" stored under `self.db_path`.

Expected Parquet schema (minimum):
- geometry: polygon/multipolygon geometry compatible with DuckDB Spatial
- code: land-use code (categorical or integer)
- bbox or extent columns used for coarse prefiltering; either as struct fields
  `bbox.xmin`, `bbox.xmax`, `bbox.ymin`, `bbox.ymax`, or equivalent flattened
  columns (e.g., `xmin`, `xmax`, `ymin`, `ymax`).

Public API:
- `LanduseCalculator.calculate_landuse_area_ratio()`

Internal helpers:
- `query_landuse_area_ratio()` reads a chunk and returns stats for one year
- `landuse_area_ratio_worker()` worker loop that processes chunks
"""
import pandas as pd
import queue
import multiprocessing as mp
from typeguard import typechecked
from typing import Self
from duckdb import DuckDBPyConnection
from tqdm import tqdm
from pathlib import Path

import duckpipe.common as C
from duckpipe.duckdb_utils import generate_duckdb_memory_connection

VALID_YEARS = [2000, 2005, 2010, 2015, 2020]
VAR_PREFIX = "LS"


def query_landuse_area_ratio(chunk: pd.DataFrame,
                             year: int,
                             buffer_sizes: list[float],
                             table_path: str | Path,
                             conn: DuckDBPyConnection,
                             ) -> pd.DataFrame:
    """
    [description]
    Compute land-use area (`a`) and proportion (`p`) within buffer rings around
    each input geometry for a single `year`, scanning the Parquet dataset at
    `table_path`.

    [input]
    - chunk: pandas.DataFrame — A chunk with columns [`C.ID_COL`, "wkt"].
    - year: int — Target year, used to choose the Parquet file (`landuse_{year}.parquet`).
    - buffer_sizes: list[float] — Buffer distances (meters) to compute.
    - table_path: str | pathlib.Path — Path to the yearly land-use Parquet file.
    - conn: duckdb.DuckDBPyConnection — In-memory DuckDB connection with Spatial loaded.

    [output]
    - pandas.DataFrame — Columns [`C.ID_COL`, `C.VAR_COL`, `C.YEAR_COL`, `C.VAL_COL`].

    [notes]
    - The function constructs an AOI via buffer of the chunk geometries for coarse prefiltering.
    - It builds variable names like `LS{lu_code}_{buffer_size:04d}_{stat_type}` where stat_type is `a` or `p`.
    """
    # prepare
    table_path = table_path
    # generate chunk table
    conn.register('chunk_wkt', chunk)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE chunk AS (
        SELECT {C.ID_COL}, ST_GeomFromText(wkt) AS geometry
        FROM chunk_wkt
    )
    """)
    # define aoi bbox
    # by duckdb optimizer error, bbox filter is faster than ST_Intersects
    # if using aoi directly in query, memory usage goes up to 9GB per worker
    # full table-scanning for every iteration (checked by EXPLAIN ANALYZE)
    max_buffer_size = max(buffer_sizes)
    sql = f"""
    WITH aoi AS ( 
        SELECT ST_Envelope(ST_Buffer(ST_Union_Agg(geometry), {max_buffer_size})) AS aoi 
        FROM chunk 
    )
    SELECT ST_XMin(aoi), ST_YMin(aoi), ST_XMax(aoi), ST_YMax(aoi), ST_AsText(aoi)
    FROM aoi
    """
    xmin, ymin, xmax, ymax, aoi_wkt = conn.execute(sql).fetchone()
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE aoi_landuse AS (
        SELECT 
            ST_Intersection(geometry, ST_GeomFromText('{aoi_wkt}')) AS geometry,
            code
        FROM '{table_path}'
        WHERE 
            xmin < {xmax} AND 
            xmax > {xmin} AND 
            ymin < {ymax} AND 
            ymax > {ymin}
    );
    CREATE INDEX rtree_aoi_landuse ON aoi_landuse
    USING RTREE (geometry) WITH (max_node_capacity = 4);
    """)
    # main query
    conn.register('buffer_size', pd.DataFrame({"buffer_size": buffer_sizes}))
    result = conn.execute(f"""
    WITH 
    aoi AS (
        SELECT 
            c.{C.ID_COL} AS {C.ID_COL}
            , bs.buffer_size AS buffer_size
            , ST_Buffer(c.geometry, bs.buffer_size) AS geometry
        FROM chunk AS c CROSS JOIN buffer_size AS bs
    )
    , aggregated AS (
        SELECT
            a.{C.ID_COL} AS {C.ID_COL}
            , a.buffer_size AS buffer_size
            , CAST(l.code AS VARCHAR) AS lu_code
            , SUM( ST_Area(ST_Intersection(l.geometry, a.geometry)) ) AS a
            , SUM( ST_Area(ST_Intersection(l.geometry, a.geometry)) / ST_Area(a.geometry) ) AS p
        FROM 
            aoi_landuse AS l INNER JOIN aoi AS a ON ST_Intersects(l.geometry, a.geometry)
        GROUP BY a.{C.ID_COL}, a.buffer_size, l.code
    )
    , unpivoted AS (
        SELECT *
        FROM aggregated
        UNPIVOT ( val FOR stat_type IN (a, p) )
    )
    , renamed AS (
        SELECT 
            {C.ID_COL}
            , CONCAT( 
                '{VAR_PREFIX}', lu_code, '_', 
                LPAD(buffer_size::VARCHAR, 4, '0'), '_', 
                stat_type 
            ) AS {C.VAR_COL}
            , {year} AS {C.YEAR_COL}
            , val AS {C.VAL_COL}
        FROM unpivoted
    )
    SELECT * 
    FROM renamed
    """).df()
    # clear temporary table
    conn.execute("DROP INDEX IF EXISTS rtree_aoi_landuse")
    conn.execute("DROP TABLE IF EXISTS chunk")
    conn.execute("DROP TABLE IF EXISTS aoi")
    conn.execute("DROP TABLE IF EXISTS aoi_landuse")
    conn.unregister('chunk_wkt')
    conn.unregister('buffer_size')
    return result


@typechecked
def landuse_area_ratio_worker(task_queue, 
                              result_queue, 
                              year: int, 
                              buffer_sizes: list[float],
                              table_path: str | Path,
                              memory_limit: str):
    """
    [description]
    Worker loop that pulls chunks from `task_queue` and computes land-use area/ratio
    stats for the specified `year` and `buffer_sizes`, scanning the Parquet file at
    `table_path`.

    [input]
    - task_queue: multiprocessing.Queue — Provides chunk DataFrames or sentinel.
    - result_queue: multiprocessing.Queue — Receives `(chunk_len, result_df)` or sentinel.
    - year: int — Target year.
    - buffer_sizes: list[float] — Buffer distances (meters).
    - table_path: str | pathlib.Path — Path to `landuse_{year}.parquet`.
    - memory_limit: str — Passed to DuckDB memory connection.

    [output]
    - None — Side effects: places results on `result_queue` and a sentinel when done.
    """
    conn = generate_duckdb_memory_connection(memory_limit=memory_limit)
    try:
        while True:
            try:
                task = task_queue.get(timeout=0.1)
                if isinstance(task, str):
                    if task == C.SENTINEL:
                        result_queue.put(C.SENTINEL)
                        break
                chunk = task
                res = query_landuse_area_ratio(chunk, year, buffer_sizes, table_path, conn)
                result_queue.put((len(chunk), res))
            except queue.Empty:
                continue
    finally:
        conn.close()
    return


class LanduseCalculator:

    @typechecked
    def calculate_landuse_area_ratio(self,
                                     years: int | list[int], 
                                     buffer_sizes: float | list[float] | None
                                     ) -> Self:
        """
        [description]
        Calculate land-use `area` and `ratio` statistics for one or more `years` and one or more
        `buffer_sizes`. Uses multiprocessing over `self.chunks` and appends results to
        `self.result_df`.

        [input]
        - years: int | list[int] — One or more years (must be in `VALID_YEARS`).
        - buffer_sizes: float | list[float] | None — Buffer distances; if None, defaults to `[0]`.

        [output]
        - Self — Returns self for chaining. Appends rows with [`C.ID_COL`, `C.VAR_COL`,
          `C.YEAR_COL`, `C.VAL_COL`].

        [example usage]
        ```python
        # from test/example.py
        geovariable = (
            calculator
            .set_dataframe(gdf)
            .chunk_by_centroid(max_cluster_size=MAX_CLUSTER_SIZE, distance_threshold=MAX_CLUSTER_WIDTH)
            .calculate_landuse_area_ratio(years=years, buffer_sizes=[100, 300, 500, 1000, 5000])
            .get_result(pivot=True)
        )
        ```
        
        [notes]
        - Data source: Parquet file per-year resolved as `(self.db_path / f"landuse_{year}").with_suffix(".parquet")`.
        - Variable naming: `LS{lu_code}_{buffer_size:04d}_{stat}` where `stat` ∈ {`a`, `p`}.
        - Results are appended to `self.result_df` in long format and can be pivoted via `get_result(pivot=True)`.
        """
        # input conversion
        if isinstance(years, int):
            years = [years]
        if isinstance(buffer_sizes, float):
            buffer_sizes = [buffer_sizes]
        if buffer_sizes is None:
            buffer_sizes = [0]
        years = sorted(years)
        buffer_sizes = sorted(buffer_sizes)
        # input check
        is_valid_year = [year in VALID_YEARS for year in years]
        if not all(is_valid_year):
            raise ValueError(f"Invalid year. Valid years are: {VALID_YEARS}")
        # perform calculations
        results = []
        for year in years:
            # multiprocessing setup (spawn-safe)
            mp.set_start_method("spawn", force=True)
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            workers = []
            table_path = (self.db_path / f"landuse_{year}").with_suffix(f".parquet")
            for _ in range(self.n_workers):
                args = (task_queue, result_queue, year, buffer_sizes, table_path, self.memory_limit)
                p = mp.Process(target=landuse_area_ratio_worker, args=args)
                p.start()
                workers.append(p)
            # enqueue chunk tasks
            for chunk in self.chunks:
                task_queue.put(chunk)
            # signal end
            for _ in range(self.n_workers):
                task_queue.put(C.SENTINEL)
            # aggregate results with progress by chunk size
            desc = f"landuse ({year}) (buffer_sizes: {buffer_sizes})"
            tq = tqdm(total=len(self.geom_df), bar_format=C.TQDM_BAR_FORMAT, desc=desc, disable=not self.verbose)
            n_alive_workers = self.n_workers
            while n_alive_workers > 0:
                result = result_queue.get()
                if isinstance(result, str):
                    if result == C.SENTINEL:
                        n_alive_workers -= 1
                else:
                    chunk_len, df = result
                    tq.update(chunk_len)
                    results.append(df)
            tq.close()
            for p in workers:
                p.join()
        # result
        df = pd.concat(results)    
        self.result_df = pd.concat([self.result_df, df], ignore_index=True)
        return self