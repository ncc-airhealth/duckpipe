"""
Parallel worker utilities to execute DuckDB Spatial SQL per geometry chunk using
multiprocessing. Each worker registers the input as a temp `chunk` table and returns
partial results to be aggregated into `self.result_df`.
"""
import pandas as pd
import multiprocessing as mp
import queue
from typeguard import typechecked
from typing import Self, Callable
from tqdm import tqdm
from duckpipe.common import SENTINEL, TQDM_BAR_FORMAT
from duckpipe.duckdb_utils import generate_duckdb_memory_connection

@typechecked
def _worker(task_queue: mp.Queue, result_queue: mp.Queue, conn_query: str="") -> None:
    """
    Worker process target that:
    - Opens an in-memory DuckDB connection and optionally runs `conn_query`.
    - Receives tasks `(input_df, pre_query, main_query, post_query)` from `task_queue`.
    - Registers `input_df` as temp table `chunk(id, geometry)` via WKT from `input_df`.
    - Executes `pre_query`, `main_query`, `post_query` in order and sends result DataFrame.
    - On sentinel (`main_query == SENTINEL`), forwards sentinel to `result_queue` and exits.

    - task_queue: mp.Queue — Queue producing work items.
    - result_queue: mp.Queue — Queue to send back partial results or sentinel.
    - conn_query: str — Optional SQL executed once per worker on its connection.

    - None
    """
    # generate duckdb connection
    conn = generate_duckdb_memory_connection()
    if conn_query:
        conn.execute(conn_query)
    # work
    while True:
        try:
            input_df, pre_query, main_query, post_query = task_queue.get(timeout=0.1)
            # stop
            if main_query == SENTINEL:
                result_queue.put(SENTINEL)
                break
            # work
            else:
                conn.register('input', input_df)
                conn.execute(f"""
                    CREATE OR REPLACE TEMP TABLE chunk AS (
                        SELECT id, ST_GeomFromText(wkt) AS geometry
                        FROM input
                    )
                """)
                conn.execute(pre_query)
                result = conn.execute(main_query).df()
                result_queue.put(obj=(len(input_df), result))
                conn.execute(post_query)
                conn.unregister('input')
                conn.execute("DROP TABLE IF EXISTS chunk")
        except queue.Empty:
            continue
    # close connection
    conn.close()
    return


class Worker:
    """
    Mixin that runs per-chunk DuckDB SQL in parallel and appends concatenated
    results to `self.result_df`.
    """
    @typechecked
    def run_query_workers(self, pre_query: str, main_query: str, post_query: str="", desc: str="") -> Self:
        """
        Run DuckDB SQL over `self.chunks` using multiple worker processes. Each worker
        creates `chunk(id, geometry)` from the input DataFrame and executes the provided
        SQL segments, returning a partial result DataFrame. Progress is tracked by `tqdm`.

        - pre_query: str — SQL executed before `main_query` per chunk (e.g., macros, temp tables).
        - main_query: str — SQL selecting rows with columns [`id`, `varname`, `year`, `value`].
        - post_query: str — Optional SQL executed after `main_query` per chunk (cleanup).
        - desc: str — Progress bar description.

        - Self — Appends to `self.result_df` and returns self.

        ```python
        pre, main, post = "", "SELECT id, 'Var' AS varname, NULL AS year, 0 AS value FROM chunk", ""
        calculator.run_query_workers(pre, main, post, desc="Example")
        ```
        """
        # prepare quque
        mp.set_start_method("spawn", force=True)
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        # generate workers
        workers = []
        for _ in range(self.n_workers):
            kwargs = dict(
                task_queue=task_queue, 
                result_queue=result_queue
            )
            p = mp.Process(target=_worker, kwargs=kwargs)
            p.start()
            workers.append(p)
        # enqueue tasks
        for chunk in self.chunks:
            task_queue.put(obj=(chunk, pre_query, main_query, post_query))
        # enqueue sentinel signal
        for _ in range(self.n_workers):
            task_queue.put(obj=(None, None, SENTINEL, None))
        # aggregate results
        tq = tqdm(total=len(self.wkt_df), bar_format=TQDM_BAR_FORMAT, desc=desc, disable=not self.verbose)
        n_alive_workers = self.n_workers
        results = []
        while n_alive_workers > 0:
            result = result_queue.get()
            if isinstance(result, str) and result == SENTINEL:
                n_alive_workers -= 1
            else:
                chunk_len, df = result
                results.append(df)
                tq.update(chunk_len)
        tq.close()
        for p in workers:
            p.join()
        # result
        df = pd.concat(results)
        self.result_df = pd.concat([self.result_df, df], ignore_index=True)
        # done
        return self