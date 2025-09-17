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
VALID_MR_TYPES = ["mr1", "mr2"]
VAR_NAME_MACRO = """
CREATE OR REPLACE MACRO varname(mr_type) AS
    CASE
        WHEN mr_type = 'mr1' THEN 'D_MR1'
        WHEN mr_type = 'mr2' THEN 'D_MR2'
        ELSE 'error_processing_main_road_distance'
    END
"""

def TQDM_DESC(mr_type, year):
    return f"{mr_type} distance ({year})"


def _query(chunk: pd.DataFrame,
           mr_type: str,
           year: int,
           table_path: str | Path,
           conn: DuckDBPyConnection) -> pd.DataFrame:
    """duckdb SQL query"""
    # create varname macro
    conn.execute(VAR_NAME_MACRO)
    # register chunk geometries
    conn.register('chunk_wkt', chunk)
    conn.execute(f"""
    CREATE OR REPLACE TEMP TABLE chunk AS (
        SELECT 
            {C.ID_COL}, 
            ST_GeomFromText(wkt) AS geometry
        FROM chunk_wkt
    )
    """)
    # compute per-id minimum distance to airports in the given year
    query = f"""
    WITH mr_sel_year AS (
        SELECT geometry AS geometry
        FROM '{table_path}'
        WHERE year = {year}
    )
    SELECT 
        c.{C.ID_COL} AS {C.ID_COL}, 
        varname('{mr_type}') AS {C.VAR_COL}, 
        {year} AS {C.YEAR_COL}, 
        MIN(ST_Distance(m.geometry, c.geometry)) AS {C.VAL_COL}
    FROM 
        chunk AS c
    CROSS JOIN 
        mr_sel_year AS m
    GROUP BY 
        c.{C.ID_COL}
    """
    result = conn.execute(query).df()
    # clear temp objects
    conn.execute("DROP TABLE IF EXISTS chunk")
    conn.unregister('chunk_wkt')
    return result


@typechecked
def _worker(task_queue: mp.Queue, 
            result_queue: mp.Queue, 
            mr_type: str, 
            year: int,
            table_path: str | Path,
            memory_limit: str):
    """worker loop"""
    conn = generate_duckdb_memory_connection(memory_limit=memory_limit)
    try:
        while True:
            try:
                task = task_queue.get(timeout=0.1)
                if isinstance(task, str) and task == C.SENTINEL:
                    result_queue.put(C.SENTINEL)
                    break
                chunk = task  # pandas DataFrame with [ID_COL, wkt]
                table_path = table_path
                result = _query(
                    chunk=chunk,
                    mr_type=mr_type,
                    year=year,
                    table_path=table_path,
                    conn=conn
                )
                result_queue.put((len(chunk), result))
            except queue.Empty:
                continue
    finally:
        conn.close()
    return


class MainRoadDistanceCalculator:

    @typechecked
    def calculate_main_road_distance(self, mr_types: str | list[str], years: int | list[int]) -> Self:
        # input conversion
        if isinstance(mr_types, str):
            mr_types = [mr_types]
        mr_types = sorted(mr_types)
        if isinstance(years, int):
            years =[years]
        years = sorted(years)
        # input check
        is_valid_mr_type = [mr_type in VALID_MR_TYPES for mr_type in mr_types]
        if not all(is_valid_mr_type):
            raise ValueError(f"Invalid mr_type. Valid mr_types are: {VALID_MR_TYPES}")
        is_valid_year = [year in VALID_YEARS for year in years]
        if not all(is_valid_year):
            raise ValueError(f"Invalid year. Valid years are: {VALID_YEARS}")
        # perform calculations
        results = []
        for mr_type in mr_types:
            for year in years:
                # multiprocessing setup (spawn-safe)
                mp.set_start_method("spawn", force=True)
                task_queue = mp.Queue()
                result_queue = mp.Queue()
                workers = []
                for _ in range(self.n_workers):
                    table = mr_type
                    table_path = (self.db_path / table).with_suffix(f".parquet")
                    args = (task_queue, result_queue, mr_type, year, table_path, self.memory_limit)
                    p = mp.Process(target=_worker, args=args)
                    p.start()
                    workers.append(p)
                # enqueue chunk tasks
                for chunk in self.chunks:
                    task_queue.put(chunk)
                # signal end
                for _ in range(self.n_workers):
                    task_queue.put(C.SENTINEL)
                # aggregate results with progress by chunk size
                description = TQDM_DESC(mr_type, year)
                tq = tqdm(total=len(self.wkt_df), bar_format=C.TQDM_BAR_FORMAT, desc=description, disable=not self.verbose)
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