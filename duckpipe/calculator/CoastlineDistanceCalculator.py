import pandas as pd
import queue
import multiprocessing as mp
from typeguard import typechecked
from typing import Self
from duckdb import DuckDBPyConnection
from tqdm import tqdm

import duckpipe.common as C
from duckpipe.duckdb_utils import generate_duckdb_connection

VALID_YEARS = [2000, 2005, 2010, 2015, 2020]
TABLE_NAME = "coastline"
VAR_PREFIX = "D_Coast"

def query_coastline_distance_chunk(chunk: pd.DataFrame, 
                                   year: int, 
                                   conn: DuckDBPyConnection) -> pd.DataFrame:
    # register chunk
    conn.register('chunk_wkt', chunk)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE chunk AS (
        SELECT {C.ID_COL}, ST_GeomFromText(wkt) AS geometry
        FROM chunk_wkt
    )
    """)
    # compute distances
    result = conn.execute(f"""
    SELECT 
        c.{C.ID_COL} AS {C.ID_COL}
        , '{VAR_PREFIX}' AS {C.VAR_COL}
        , {year} AS {C.YEAR_COL}
        , MIN(ST_Distance(t.geometry, c.geometry)) AS {C.VAL_COL}
    FROM chunk AS c
    CROSS JOIN {TABLE_NAME} AS t
    WHERE t.year = {year}
    GROUP BY c.{C.ID_COL}
    """).df()
    # clear
    conn.execute("DROP TABLE IF EXISTS chunk")
    conn.unregister('chunk_wkt')
    return result


@typechecked
def coastline_distance_worker(task_queue, 
                              result_queue, 
                              db_path: str, 
                              year: int,
                              memory_limit: str):
    conn = generate_duckdb_connection(db_path, memory_limit=memory_limit)
    try:
        while True:
            try:
                task = task_queue.get(timeout=0.1)
                if isinstance(task, str) and task == C.SENTINEL:
                    result_queue.put(C.SENTINEL)
                    break
                chunk = task
                result = query_coastline_distance_chunk(chunk, year, conn)
                result_queue.put((len(chunk), result))
            except queue.Empty:
                continue
    finally:
        conn.close()
    return


class CoastlineDistanceCalculator:

    @typechecked
    def calculate_coastline_distance(self, years: int | list[int]) -> Self:
        """
        [description]
        Calculate per-feature minimum distance to the coastline for one or more years. Uses
        multiprocessing over precomputed `self.chunks` and appends results to `self.result_df`.

        [input]
        - years: int | list[int] — One or more target years (must be in `VALID_YEARS`).

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
            .calculate_coastline_distance(years=years)
            .get_result(pivot=True)
        )
        ```
        """
        # input conversion
        if isinstance(years, int):
            years = [years]
        years = sorted(years)
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
            for _ in range(self.n_workers):
                args = (task_queue, result_queue, self.db_path, year, self.memory_limit)
                p = mp.Process(target=coastline_distance_worker, args=args)
                p.start()
                workers.append(p)
            # enqueue chunk tasks
            for chunk in self.chunks:
                task_queue.put(chunk)
            # signal end
            for _ in range(self.n_workers):
                task_queue.put(C.SENTINEL)
            # aggregate results with progress by chunk size
            description = f"{VAR_PREFIX} ({year})"
            tq = tqdm(total=len(self.geom_df), bar_format=C.TQDM_BAR_FORMAT, desc=description, disable=not self.verbose)
            n_alive_workers = self.n_workers
            while n_alive_workers > 0:
                result = result_queue.get()
                if isinstance(result, str) and result == C.SENTINEL:
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
