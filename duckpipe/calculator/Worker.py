"""
Parallel worker utilities to execute DuckDB Spatial SQL per geometry chunk using
multiprocessing. Each worker registers the input as a temp `chunk` table and returns
partial results to be aggregated into `self.result_df`.
"""
import pandas as pd
import multiprocessing as mp
import queue
from typeguard import typechecked
from typing import Self
from tqdm import tqdm
from duckpipe.common import SENTINEL, TQDM_BAR_FORMAT
from duckpipe.duckdb_utils import generate_duckdb_memory_connection
from enum import Enum
from duckdb import DuckDBPyConnection
from datetime import datetime

class WorkerMode(Enum):
    CHUNKED_MULTI = "chunked_multi"
    CHUNKED_SINGLE = "chunked_single"
    TOTAL_SINGLE = "total_single"



def _run_query(conn: DuckDBPyConnection, 
               input_df: pd.DataFrame, 
               pre_query: str, 
               main_query: str, 
               post_query: str) -> pd.DataFrame:
    """Run DuckDB SQL over `input_df` using a single worker process."""
    # work
    conn.register('input', input_df)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE chunk AS (
            SELECT id, ST_GeomFromText(wkt) AS geometry
            FROM input
        )
    """)
    conn.execute(pre_query)
    result = conn.execute(main_query).df()
    # raise Exception(conn.execute("SELECT * FROM chunk").df()) # line for query debugging
    conn.execute(post_query)
    conn.unregister('input')
    conn.execute("DROP TABLE IF EXISTS chunk")
    # done
    return result

@typechecked
def _worker(task_queue: mp.Queue, result_queue: mp.Queue, conn_query: str="") -> None:
    """
    [description]
    Worker process target that opens an in-memory DuckDB connection and optionally runs `conn_query`.
    Receives tasks `(input_df, pre_query, main_query, post_query)` from `task_queue`, registers `input_df` as temp table `chunk(id, geometry)` via WKT from `input_df`, executes `pre_query`, `main_query`, `post_query` in order, and sends result DataFrame.
    On sentinel (`main_query == SENTINEL`), forwards sentinel to `result_queue` and exits.

    [input]
    - task_queue: mp.Queue — Queue producing work items.
    - result_queue: mp.Queue — Queue to send back partial results or sentinel.
    - conn_query: str — Optional SQL executed once per worker on its connection (default: "").

    [output]
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
                result = _run_query(conn, input_df, pre_query, main_query, post_query)
                result_queue.put(obj=(len(input_df), result))
        except queue.Empty:
            continue
    # close connection
    conn.close()
    return


class Worker:
    """
    [description]
    Mixin that executes DuckDB Spatial SQL over geometry chunks in parallel or single-process modes.
    Provides a unified runner that registers each input chunk as a temp table `chunk(id, geometry)`,
    executes user-provided `pre_query`, `main_query`, and `post_query`, and appends results to
    `self.result_df`.

    [input]
    - None (This is a mixin). Implementations are expected to provide the following attributes:
      - n_workers: int — Degree of parallelism or DuckDB threads depending on mode.
      - chunks: list[pd.DataFrame] — List of input DataFrame chunks with columns [`id`, `wkt`].
      - wkt_df: pd.DataFrame — Full input DataFrame used for progress bar and TOTAL_SINGLE mode.
      - result_df: pd.DataFrame — Accumulator DataFrame to which results are appended.
      - verbose: bool — Whether to display progress bars / extra logs.
      - worker_mode: WorkerMode — Default execution mode.

    [output]
    - None
    """
    @typechecked
    def run_query_workers(self, 
                          pre_query: str, 
                          main_query: str, 
                          post_query: str, 
                          mode: WorkerMode | str=WorkerMode.CHUNKED_MULTI, 
                          desc: str="") -> Self:
        """
        [description]
        Dispatch method to run the provided DuckDB SQL over geometry chunks using the selected mode.
        Converts `mode` to `WorkerMode` if given as a string and calls the corresponding runner.

        [input]
        - pre_query: str — SQL executed before `main_query` (e.g., temp tables, indices, macros).
        - main_query: str — SQL that returns a DataFrame with rows to be appended to `self.result_df`.
        - post_query: str — SQL executed after `main_query` (e.g., cleanup of temp objects).
        - mode: WorkerMode | str — One of: `CHUNKED_MULTI`, `CHUNKED_SINGLE`, `TOTAL_SINGLE`.
        - desc: str — Optional description shown in the progress bar / logs.

        [output]
        - Self — Returns self for method chaining.
        """
        if isinstance(mode, str):
            mode = WorkerMode(mode)
        if mode == WorkerMode.CHUNKED_MULTI:
            self._run_chunked_multi(pre_query, main_query, post_query, desc)
        elif mode == WorkerMode.CHUNKED_SINGLE:
            self._run_chunked_single(pre_query, main_query, post_query, desc)
        elif mode == WorkerMode.TOTAL_SINGLE:
            self._run_total_single(pre_query, main_query, post_query, desc)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return self

    @typechecked
    def _run_chunked_multi(self, 
        pre_query: str, 
        main_query: str, 
        post_query: str, 
        desc: str="", 
        conn_query: str=""
    ) -> Self:
        """
        [description]
        Run DuckDB SQL over `self.chunks` using multiple worker processes. Uses queues to distribute
        tasks to workers and a sentinel to signal completion. Each worker registers its chunk as
        `chunk(id, geometry)` from WKT and executes `pre_query`, `main_query`, `post_query` in order.
        Partial results are concatenated and appended to `self.result_df`.

        [input]
        - pre_query: str — SQL executed before `main_query` per worker task.
        - main_query: str — SQL returning the partial result for a chunk.
        - post_query: str — SQL executed after `main_query` per worker task.
        - desc: str — Description for the progress bar.
        - conn_query: str — SQL executed once per worker on its connection (e.g., PRAGMAs, macros).

        [output]
        - Self — Returns self after appending concatenated results to `self.result_df`.
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
                result_queue=result_queue, 
                conn_query=conn_query
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
    
    @typechecked
    def _run_chunked_single(self, 
                            pre_query: str, 
                            main_query: str, 
                            post_query: str, 
                            desc: str="", 
                            conn_query: str="") -> Self:
        """
        [description]
        Run DuckDB SQL over `self.chunks` in a single process with DuckDB configured to use
        `self.n_workers` threads. Processes chunks sequentially on one connection and concatenates
        all partial results into `self.result_df`.

        [input]
        - pre_query: str — SQL executed before `main_query` per chunk.
        - main_query: str — SQL returning the partial result for a chunk.
        - post_query: str — SQL executed after `main_query` per chunk.
        - desc: str — Description for the progress bar.
        - conn_query: str — SQL executed once on the connection.

        [output]
        - Self — Returns self after appending concatenated results to `self.result_df`.
        """
        # generate duckdb connection
        conn = generate_duckdb_memory_connection()
        if conn_query:
            conn.execute(conn_query)
        conn.execute(f"SET threads = {self.n_workers};")
        # work
        tq = tqdm(total=len(self.wkt_df), bar_format=TQDM_BAR_FORMAT, desc=desc, disable=not self.verbose)
        results = []
        for chunk in self.chunks:
            result = _run_query(conn, chunk, pre_query, main_query, post_query)
            tq.update(len(chunk))
            results.append(result)
        tq.close()
        # close connection
        conn.close()
        # result
        df = pd.concat(results)
        self.result_df = pd.concat([self.result_df, df], ignore_index=True)
        # done
        return self
    
    @typechecked
    def _run_total_single(self, 
                          pre_query: str, 
                          main_query: str, 
                          post_query: str, 
                          desc: str="", 
                          conn_query: str="") -> Self:
        """
        [description]
        Run DuckDB SQL once over the entire input `self.wkt_df` on a single in-memory connection.
        Configures DuckDB threads and optional progress bar. Appends the full result to
        `self.result_df`.

        [input]
        - pre_query: str — SQL executed before `main_query` (global scope).
        - main_query: str — SQL returning the final result DataFrame for the whole dataset.
        - post_query: str — SQL executed after `main_query` (global scope).
        - desc: str — Text printed before/after to indicate start and completion.
        - conn_query: str — SQL executed once on the connection.

        [output]
        - Self — Returns self after appending the result to `self.result_df`.
        """
        # generate duckdb connection
        conn = generate_duckdb_memory_connection()
        if conn_query:
            conn.execute(conn_query)
        conn.execute(f"SET enable_progress_bar = {self.verbose};")
        conn.execute(f"SET threads = {self.n_workers};")
        # work
        start_time = datetime.now().isoformat(timespec="seconds")
        print(f"[{start_time}] {desc} started")
        df = _run_query(conn, self.wkt_df, pre_query, main_query, post_query)
        self.result_df = pd.concat([self.result_df, df], ignore_index=True)
        # done
        end_time = datetime.now().isoformat(timespec="seconds")
        print(f"[{end_time}] {desc} completed")
        conn.close()
        return self