"""
Relative elevation calculator (Parquet backend).

Computes reference elevation at features and relative elevation ratios in donut
rings for requested elevation sources (e.g., DEM/DSM) and buffer sizes. Uses
multiprocessing with an in-memory DuckDB + Spatial connection. Elevation data
is read from Parquet files named "dem.parquet" and "dsm.parquet" under
`self.db_path`.

Expected Parquet schema (minimum):
- geometry: pixel/cell footprint polygons compatible with DuckDB Spatial
- value: numeric elevation value per pixel/cell
- centroid_x, centroid_y: numeric columns used for coarse AOI prefiltering

Public API:
- `RelativeElevationCalculator.calculate_relative_elevation()`

Internal helpers:
- `query_relative_elevation_chunk()` reads a chunk and returns stats for one source
- `relative_elevation_worker()` worker loop that processes chunks
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

VALID_TABLE_VAR_NAME = {
    "dem": ("Alt_k", "Altitude_k"), 
    "dsm": ("Alt_a", "Altitude_a")
}
DONUT_THICKNESS = 30
DEM_SPATIAL_RESOLUTION = 90


def query_relative_elevation_chunk(chunk: pd.DataFrame,
                                   table: str,
                                   buffer_sizes: list[float], 
                                   table_path: str | Path,
                                   conn: DuckDBPyConnection
                                   ) -> pd.DataFrame:
    """
    [description]
    Compute reference elevation at features and relative elevation ratios within
    donut rings for one elevation source (`table`) and the given `buffer_sizes`,
    scanning the Parquet dataset at `table_path`.

    [input]
    - chunk: pandas.DataFrame — A chunk with columns [`C.ID_COL`, "wkt"].
    - table: str — Elevation source name (e.g., "dem", "dsm").
    - buffer_sizes: list[float] — Buffer distances (meters) for donut rings.
    - table_path: str | pathlib.Path — Filesystem path to the Parquet file for `table`.
    - conn: duckdb.DuckDBPyConnection — In-memory DuckDB connection with Spatial loaded.

    [output]
    - pandas.DataFrame — Long-form rows for relative elevation ratios and reference elevation
      with columns [`C.ID_COL`, `C.VAR_COL`, `C.YEAR_COL` (NULL), `C.VAL_COL`].
    """
    # prepare
    rel_elev_prefix, ref_elev_prefix = VALID_TABLE_VAR_NAME[table]
    # generate chunk table
    conn.register('chunk_wkt', chunk)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE chunk AS (
        SELECT 
            {C.ID_COL}, 
            ST_GeomFromText(wkt) AS geometry
        FROM chunk_wkt
    )
    """)
    # query elevation subset
    max_buffer_size = max(buffer_sizes)
    clip_distance = max_buffer_size + DONUT_THICKNESS + 2 * DEM_SPATIAL_RESOLUTION
    query = f"""
    SELECT 
        MIN(ST_XMin(geometry)) - {clip_distance} AS xmin, 
        MIN(ST_YMin(geometry)) - {clip_distance} AS ymin, 
        MAX(ST_XMax(geometry)) + {clip_distance} AS xmax, 
        MAX(ST_YMax(geometry)) + {clip_distance} AS ymax
    FROM 
        chunk
    """
    xmin, ymin, xmax, ymax = conn.execute(query).fetchone()
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE aoi_elevation AS (
        SELECT 
            value AS val, 
            geometry
        FROM 
            '{table_path}'
        WHERE 
            (centroid_x < {xmax}) AND 
            (centroid_x > {xmin}) AND 
            (centroid_y < {ymax}) AND 
            (centroid_y > {ymin})
    );
    CREATE INDEX rtree_temp ON aoi_elevation
    USING RTREE (geometry)
    WITH (max_node_capacity = 4);
    """)
    # value
    conn.register('buffer_size', pd.DataFrame({"buffer_size": buffer_sizes}))
    result = conn.execute(f"""
    WITH 
    ref_elevation AS (
        SELECT 
            c.{C.ID_COL} 
            , MEAN(a.val) AS ref_val -- if point touches multiple pixels
        FROM 
            chunk AS c
        LEFT JOIN 
            aoi_elevation AS a 
            ON ST_Intersects(c.geometry, a.geometry)
        GROUP BY 
            c.{C.ID_COL}
    )
    , donut AS (
        SELECT 
            c.{C.ID_COL}, 
            bs.buffer_size, 
            re.ref_val, 
            ST_Difference(
                ST_Buffer(c.geometry, bs.buffer_size + {DONUT_THICKNESS}), 
                ST_Buffer(c.geometry, bs.buffer_size)
            ) AS geometry
        FROM 
            chunk AS c
        CROSS JOIN 
            buffer_size AS bs
        INNER JOIN 
            ref_elevation AS re 
            ON c.{C.ID_COL} = re.{C.ID_COL}
    )
    , rel_elevation_ratio AS (
        SELECT 
            d.{C.ID_COL}, 
            d.buffer_size, 
            AVG(CAST((a.val - d.ref_val) > +20 AS INT)) AS above_20,
            AVG(CAST((a.val - d.ref_val) < -20 AS INT)) AS below_20,
            AVG(CAST((a.val - d.ref_val) > +50 AS INT)) AS above_50,
            AVG(CAST((a.val - d.ref_val) < -50 AS INT)) AS below_50
        FROM 
            aoi_elevation AS a
        INNER JOIN 
            donut AS d 
            ON ST_Intersects(d.geometry, a.geometry)
        GROUP BY 
            d.{C.ID_COL}, 
            d.buffer_size
    )
    , unpivoted AS (
        SELECT *
        FROM rel_elevation_ratio
        UNPIVOT ( val FOR stat IN (above_20, below_20, above_50, below_50) )
    )
    , result_rel_elev AS (
        SELECT 
            {C.ID_COL}, 
            CONCAT( '{rel_elev_prefix}_', stat, '_', buffer_size::VARCHAR ) AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            val AS {C.VAL_COL}
        FROM unpivoted
    )
    , result_ref_elev AS (
        SELECT 
            {C.ID_COL}, 
            '{ref_elev_prefix}' AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            ref_val AS {C.VAL_COL}
        FROM ref_elevation
    )
    , result AS (
        SELECT * FROM result_rel_elev
        UNION ALL
        SELECT * FROM result_ref_elev
    )
    SELECT * FROM result
    """).df()
    # clear temporary table
    conn.execute("DROP INDEX rtree_temp")
    conn.execute("DROP TABLE IF EXISTS chunk")
    conn.execute("DROP TABLE IF EXISTS aoi")
    conn.execute("DROP TABLE IF EXISTS aoi_elevation")
    conn.unregister('chunk_wkt')
    conn.unregister('buffer_size')
    # done
    return result


@typechecked
def relative_elevation_worker(task_queue,
                              result_queue,
                              table: str,
                              buffer_sizes: list[float], 
                              table_path: str | Path,
                              memory_limit: str = "4GB"):
    """
    [description]
    Worker loop that pulls chunks from `task_queue` and computes relative elevation
    metrics for the specified elevation `table` and `buffer_sizes`, scanning the
    Parquet file at `table_path`.

    [input]
    - task_queue: multiprocessing.Queue — Provides chunk DataFrames or sentinel.
    - result_queue: multiprocessing.Queue — Receives `(chunk_len, result_df)` or sentinel.
    - table: str — Elevation source (e.g., "dem", "dsm").
    - buffer_sizes: list[float] — Buffer distances (meters) for donut rings.
    - table_path: str | pathlib.Path — Path to Parquet file for the elevation source.
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
                chunk = task  # pandas DataFrame [ID_COL, wkt]
                res = query_relative_elevation_chunk(chunk, table, buffer_sizes, table_path, conn)
                result_queue.put((len(chunk), res))
            except queue.Empty:
                continue
    finally:
        conn.close()
    return


class RelativeElevationCalculator:

    @typechecked
    def calculate_relative_elevation(self, 
                                     elevation_types: str | list[str], 
                                     buffer_sizes: float | list[float]
                                     ) -> Self:
        """
        [description]
        Calculate relative elevation metrics for one or more `elevation_types` (e.g., "dem",
        "dsm") and `buffer_sizes`. Uses multiprocessing over `self.chunks` and appends rows to
        `self.result_df`.

        [input]
        - elevation_types: str | list[str] — Elevation sources to compute. Must be in
          `VALID_TABLE_VAR_NAME.keys()`.
        - buffer_sizes: float | list[float] — Buffer distances for donut rings.

        [output]
        - Self — Returns self for chaining. Appends rows with [`C.ID_COL`, `C.VAR_COL`,
          `C.YEAR_COL` (NULL), `C.VAL_COL`].

        [example usage]
        ```python
        # from test/example.py
        geovariable = (
            calculator
            .set_dataframe(gdf)
            .chunk_by_centroid(max_cluster_size=MAX_CLUSTER_SIZE, distance_threshold=MAX_CLUSTER_WIDTH)
            .calculate_relative_elevation(elevation_types=["dem", "dsm"], buffer_sizes=[1000, 5000])
            .get_result(pivot=True)
        )
        ```
        """
        # input conversion
        if isinstance(elevation_types, str):
            elevation_types = [elevation_types]
        if isinstance(buffer_sizes, float):
            buffer_sizes = [buffer_sizes]
        # input check
        valid_types = list(VALID_TABLE_VAR_NAME.keys())
        is_valid_elevation = all(et in valid_types for et in elevation_types)
        if not is_valid_elevation:
            raise ValueError(f"Invalid elevation type. Valid types are: {valid_types}")
        # perform calculations
        results = []
        for elevation_type in elevation_types:
            table = elevation_type
            # multiprocessing setup (spawn-safe)
            mp.set_start_method("spawn", force=True)
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            workers = []
            table_path = (self.db_path / table).with_suffix(f".parquet")
            for _ in range(self.n_workers):
                args = (task_queue, result_queue, table, buffer_sizes, table_path, self.memory_limit)
                p = mp.Process(target=relative_elevation_worker, args=args)
                p.start()
                workers.append(p)
            # enqueue chunk tasks
            for chunk in self.chunks:
                task_queue.put(chunk)
            # signal end
            for _ in range(self.n_workers):
                task_queue.put(C.SENTINEL)
            # aggregate results with progress by chunk size
            description = f"{VALID_TABLE_VAR_NAME[table]} (buffer_sizes: {buffer_sizes})"
            tq = tqdm(total=len(self.geom_df), bar_format=C.TQDM_BAR_FORMAT, desc=description, disable=not self.verbose)
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
