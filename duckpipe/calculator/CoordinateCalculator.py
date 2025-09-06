"""
Coordinate calculator.

Derives representative coordinates (centroid or representative point) for input
features and exports them in both WGS84 (EPSG:4326) and a projected CRS
(EPSG:5179) as scalar variables. Operates on `self.geom_df` prepared by
`Calculator.set_dataframe()` and appends results to `self.result_df`.

Public API:
- `CoordinateCalculator.calculate_coordinate()`
"""
import pandas as pd
import queue
import multiprocessing as mp
from duckdb import DuckDBPyConnection
from typeguard import typechecked
from typing import Self
from tqdm import tqdm

import duckpipe.common as C
from duckpipe.duckdb_utils import generate_duckdb_memory_connection


SUPPORTED_MODE_FUNCS = {
    "centroid": "ST_Centroid",
    "representative_point": "ST_PointOnSurface"
}
GCS_EPSG = 4326
PCS_EPSG = 5179
GCS_VAR_X = "WGS_X"
GCS_VAR_Y = "WGS_Y"
PCS_VAR_X = "TM_X"
PCS_VAR_Y = "TM_Y"


def query_coordinate_chunk(chunk: pd.DataFrame,
                           mode: str,
                           conn: DuckDBPyConnection) -> pd.DataFrame:
    """
    Compute representative point coordinates for a given chunk of features using the
    selected `mode` (centroid or representative_point).

    Parameters:
    - chunk: pandas.DataFrame — Columns [`C.ID_COL`, "wkt"].
    - mode: str — One of {"centroid", "representative_point"}.
    - conn: duckdb.DuckDBPyConnection — In-memory DuckDB connection with Spatial.

    Returns:
    - pandas.DataFrame — Long-form rows with [`C.ID_COL`, `C.VAR_COL`, `C.VAL_COL`].
    """
    coord_func = SUPPORTED_MODE_FUNCS[mode]
    # register chunk
    conn.register('chunk_wkt', chunk)
    query = f"""
    WITH 
    raw_coords AS (
        SELECT 
            {C.ID_COL}, 
            ST_Point(
                ST_Y({coord_func}(ST_GeomFromText(wkt))), -- EPSG 5179 X/Y flipped
                ST_X({coord_func}(ST_GeomFromText(wkt)))  -- EPSG 5179 X/Y flipped
            ) AS geometry
        FROM chunk_wkt
    ),
    gcs_xcoords AS (
        SELECT 
            {C.ID_COL}, 
            '{GCS_VAR_X}' AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            ST_Y( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{GCS_EPSG}') ) AS {C.VAL_COL}
        FROM raw_coords
    ),
    gcs_ycoords AS (
        SELECT 
            {C.ID_COL}, 
            '{GCS_VAR_Y}' AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            ST_X( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{GCS_EPSG}') ) AS {C.VAL_COL}
        FROM raw_coords
    ),
    pcs_xcoords AS (
        SELECT 
            {C.ID_COL}, 
            '{PCS_VAR_X}' AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            ST_Y( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{PCS_EPSG}') ) AS {C.VAL_COL}
        FROM raw_coords
    ), 
    pcs_ycoords AS (
        SELECT 
            {C.ID_COL}, 
            '{PCS_VAR_Y}' AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            ST_X( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{PCS_EPSG}') ) AS {C.VAL_COL}
        FROM raw_coords
    ),
    coords AS (
        SELECT * FROM gcs_xcoords
        UNION ALL
        SELECT * FROM gcs_ycoords
        UNION ALL
        SELECT * FROM pcs_xcoords
        UNION ALL
        SELECT * FROM pcs_ycoords
    )
    SELECT * FROM coords
    """
    df = conn.execute(query).df()
    conn.unregister('chunk_wkt')
    return df


@typechecked
def coordinate_worker(task_queue,
                      result_queue,
                      mode: str,
                      memory_limit: str):
    """
    Worker loop to compute coordinate variables for chunks using the specified `mode`.

    Parameters:
    - task_queue: multiprocessing.Queue — Provides chunk DataFrames or sentinel.
    - result_queue: multiprocessing.Queue — Receives `(chunk_len, result_df)` or sentinel.
    - mode: str — One of {"centroid", "representative_point"}.
    - memory_limit: str — Passed to DuckDB memory connection.

    Returns:
    - None — Side effects: places results on `result_queue` and a sentinel when done.
    """
    conn = generate_duckdb_memory_connection(memory_limit=memory_limit)
    try:
        while True:
            try:
                task = task_queue.get(timeout=0.1)
                if isinstance(task, str) and task == C.SENTINEL:
                    result_queue.put(C.SENTINEL)
                    break
                chunk = task  # pandas DataFrame with [ID_COL, wkt]
                res = query_coordinate_chunk(chunk, mode, conn)
                result_queue.put((len(chunk), res))
            except queue.Empty:
                continue
    finally:
        conn.close()
    return


class CoordinateCalculator:

    @typechecked
    def calculate_coordinate(self, mode: str="centroid") -> Self:
        """
        Compute representative coordinates for all features using multiprocessing
        over precomputed `self.chunks`, and append results to `self.result_df`.

        Parameters:
        - mode: str — One of {"centroid", "representative_point"}. Defaults to "centroid".

        Returns:
        - Self — Returns self for chaining. Appends rows with [`C.ID_COL`, `C.VAR_COL`, `C.VAL_COL`].
        """
        # input check
        _supported_modes = [k for k in SUPPORTED_MODE_FUNCS.keys()]
        if mode not in _supported_modes:
            raise ValueError(f"Invalid mode. Valid modes are: {_supported_modes}")
        # multiprocessing setup and execution
        results: list[pd.DataFrame] = []
        mp.set_start_method("spawn", force=True)
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        workers = []
        for _ in range(self.n_workers):
            args = (task_queue, result_queue, mode, self.memory_limit)
            p = mp.Process(target=coordinate_worker, args=args)
            p.start()
            workers.append(p)
        # enqueue chunk tasks
        for chunk in self.chunks:
            task_queue.put(chunk)
        # signal end
        for _ in range(self.n_workers):
            task_queue.put(C.SENTINEL)
        # aggregate results with progress by chunk size
        description = f"coordinate ({mode})"
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
        # done
        df = pd.concat(results) if results else pd.DataFrame(columns=[C.ID_COL, C.VAR_COL, C.VAL_COL])
        self.result_df = pd.concat([self.result_df, df], ignore_index=True)
        return self