import pandas as pd
import queue
import multiprocessing as mp
from typeguard import typechecked
from typing import Self
from duckdb import DuckDBPyConnection
from tqdm import tqdm

import duckpipe.common as C
from duckpipe.duckdb_utils import generate_duckdb_connection

VALID_TABLE_VAR_NAME = {
    "dem": "alt_k", 
    "dsm": "alt_a"
}
DONUT_THICKNESS = 30
DEM_SPATIAL_RESOLUTION = 30


def query_relative_elevation_chunk(chunk: pd.DataFrame,
                                   table: str,
                                   buffer_sizes: list[float], 
                                   conn: DuckDBPyConnection
                                   ) -> pd.DataFrame:
    var_prefix = VALID_TABLE_VAR_NAME[table]
    # generate chunk table
    conn.register('chunk_wkt', chunk)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE chunk AS (
        SELECT {C.ID_COL}, ST_GeomFromText(wkt) AS geometry
        FROM chunk_wkt
    )
    """)
    # query elevation subset
    max_buffer_size = max(buffer_sizes)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE aoi_elevation AS (
        WITH aoi AS (
            SELECT ST_Buffer(ST_Union_Agg(geometry), {max_buffer_size + DONUT_THICKNESS}) AS geometry
            FROM chunk
        )
        SELECT 
            t.val AS val
            , t.geometry AS geometry
        FROM 
            {table} AS t 
            INNER JOIN aoi AS a ON ST_Intersects(t.geometry, a.geometry)
    );
    CREATE INDEX rtree_temp ON aoi_elevation
    USING RTREE (geometry) WITH (max_node_capacity = 4);
    """)
    # value
    conn.register('buffer_size', pd.DataFrame({"buffer_size": buffer_sizes}))
    result = conn.execute(f"""
    WITH 
    -- reference elevation
    ref_elevation AS (
        SELECT 
            c.{C.ID_COL} AS {C.ID_COL}
            , MIN_BY(a.val, ST_Distance(c.geometry, a.geometry)) AS ref_val
        FROM 
            chunk AS c
            INNER JOIN aoi_elevation AS a 
                ON ST_DWithin(c.geometry, a.geometry, {DEM_SPATIAL_RESOLUTION})
        GROUP BY c.{C.ID_COL}
    )
    -- donut ring
    , donut AS (
        SELECT 
            {C.ID_COL}
            , bs.buffer_size AS buffer_size
            , ST_Difference(
                ST_Buffer(geometry, bs.buffer_size + {DONUT_THICKNESS}), 
                ST_Buffer(geometry, bs.buffer_size)
            ) AS geometry
        FROM chunk
            CROSS JOIN buffer_size AS bs
    )
    -- elevation in donut ring 
    , donut_elevation AS (
        SELECT 
            d.{C.ID_COL} AS {C.ID_COL}
            , d.buffer_size AS buffer_size
            , a.val AS elevation
        FROM 
            donut AS d
            LEFT JOIN aoi_elevation AS a ON ST_Within(a.geometry, d.geometry)
    )
    -- aggregate
    , agg AS (
        SELECT
            {C.ID_COL}
            , buffer_size
            , AVG(CAST(elevation >= 20 AS INT)) AS above_20
            , AVG(CAST(elevation <  20 AS INT)) AS below_20
            , AVG(CAST(elevation >= 50 AS INT)) AS above_50
            , AVG(CAST(elevation <  50 AS INT)) AS below_50
        FROM donut_elevation
        GROUP BY {C.ID_COL}, buffer_size
    )
    -- result
    , melted AS (
        SELECT *
        FROM agg
        UNPIVOT ( val FOR stat IN (above_20, below_20, above_50, below_50) )
    )
    , result AS (
        SELECT 
            {C.ID_COL}
            , CONCAT( '{var_prefix}_', stat, '_', buffer_size ) AS {C.VAR_COL}
            , NULL AS {C.YEAR_COL}
            , val AS {C.VAL_COL}
        FROM melted
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
    return result


@typechecked
def relative_elevation_worker(task_queue,
                              result_queue,
                              db_path: str,
                              table: str,
                              buffer_sizes: list[float], 
                              memory_limit: str = "4GB"):
    conn = generate_duckdb_connection(db_path, memory_limit=memory_limit)
    try:
        while True:
            try:
                task = task_queue.get(timeout=0.1)
                if isinstance(task, str):
                    if task == C.SENTINEL:
                        result_queue.put(C.SENTINEL)
                        break
                chunk = task  # pandas DataFrame [ID_COL, wkt]
                res = query_relative_elevation_chunk(chunk, table, buffer_sizes, conn)
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
            for _ in range(self.n_workers):
                args = (task_queue, result_queue, self.db_path, table, buffer_sizes, self.memory_limit)
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
