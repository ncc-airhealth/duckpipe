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

DONUT_THICKNESS = 30
DEM_SPATIAL_RESOLUTION = 30
VALID_ELEVATION_TYPES = ["dem", "dsm"]
VAR_NAME_MACRO_REL = """
CREATE OR REPLACE MACRO varname_rel(elev_type, stat, buffer_size) AS
    printf('%s_%s_%s', 
        CASE
            WHEN elev_type = 'dem' THEN 'Alt_k'
            WHEN elev_type = 'dsm' THEN 'Alt_a'
            ELSE 'error_processing_relative_elevation'
        END, 
        stat, 
        buffer_size::VARCHAR
    )
"""
VAR_NAME_MACRO_REF = """
CREATE OR REPLACE MACRO varname_ref(elev_type) AS 
    CASE
        WHEN elev_type = 'dem' THEN 'Altitude_k'
        WHEN elev_type = 'dsm' THEN 'Altitude_a'
        ELSE 'error_processing_relative_elevation'
    END
"""
def TQDM_DESC(elev_type, buffer_sizes):
    return f"Relative elevation ({elev_type}) (buffer_sizes: {buffer_sizes})"


def _query(chunk: pd.DataFrame,
           elevation_type: str,
           buffer_sizes: list[float], 
           table_path: str | Path,
           conn: DuckDBPyConnection
           ) -> pd.DataFrame:
    """duckdb SQL query"""
    # input check
    if elevation_type not in VALID_ELEVATION_TYPES:
        raise ValueError(f"Invalid elevation type: {elevation_type}")
    # prepare
    conn.execute(VAR_NAME_MACRO_REL)
    conn.execute(VAR_NAME_MACRO_REF)
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
    # build AOI + elevation subset (clip to AOI) with RTREE index
    max_buffer_size = max(buffer_sizes)
    clip_distance = max_buffer_size + DONUT_THICKNESS + 2 * DEM_SPATIAL_RESOLUTION
    query = f"""
    CREATE OR REPLACE TEMP TABLE aoi_elevation AS (
        WITH 
        aoi AS (
            SELECT
                MIN(ST_XMin(geometry)) - {clip_distance} AS xmin,
                MIN(ST_YMin(geometry)) - {clip_distance} AS ymin,
                MAX(ST_XMax(geometry)) + {clip_distance} AS xmax,
                MAX(ST_YMax(geometry)) + {clip_distance} AS ymax
            FROM chunk
        )
        , filtered AS (
            SELECT 
                ST_MakeEnvelope(t.xmin, t.ymin, t.xmax, t.ymax) AS geometry,
                COALESCE(t.value, 0) AS elev
            FROM '{table_path}' AS t
            INNER JOIN aoi AS a
            ON 
                t.xmin > a.xmin AND 
                t.xmax < a.xmax AND 
                t.ymin > a.ymin AND 
                t.ymax < a.ymax
        )
        SELECT elev, geometry
        FROM filtered
        WHERE NOT ST_IsEmpty(geometry)
    );
    CREATE INDEX rtree_aoi_elevation ON aoi_elevation
    USING RTREE (geometry) 
    WITH (max_node_capacity = 4);
    """
    conn.execute(query)
    _df = conn.execute("SELECT * FROM aoi_elevation").df()
    if _df.empty:
        raise ValueError("No elevation data found")
    # find reference elevation
    query = f"""
    CREATE OR REPLACE TEMP TABLE ref_elevation AS (
        SELECT 
            c.{C.ID_COL}, 
            MEAN(a.elev) AS ref_elev -- if point touches multiple pixels
        FROM 
            chunk AS c
        LEFT JOIN 
            aoi_elevation AS a 
            ON ST_Intersects(c.geometry, a.geometry)
        GROUP BY 
            c.{C.ID_COL}
    );
    SELECT 
        {C.ID_COL}, 
        varname_ref('{elevation_type}') AS {C.VAR_COL}, 
        NULL AS {C.YEAR_COL}, 
        ref_elev AS {C.VAL_COL}
    FROM ref_elevation
    """
    result_ref_elev = conn.execute(query).df()
    # find relative elevation
    conn.register('buffer_size', pd.DataFrame({"buffer_size": buffer_sizes}))
    query = f"""
    WITH 
    donut AS (
        SELECT 
            c.{C.ID_COL}, 
            bs.buffer_size, 
            re.ref_elev, 
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
            AVG(CAST((a.elev - d.ref_elev) > +20 AS INT)) AS above_20,
            AVG(CAST((a.elev - d.ref_elev) < -20 AS INT)) AS below_20,
            AVG(CAST((a.elev - d.ref_elev) > +50 AS INT)) AS above_50,
            AVG(CAST((a.elev - d.ref_elev) < -50 AS INT)) AS below_50
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
            varname_rel('{elevation_type}', stat, buffer_size) AS {C.VAR_COL}, 
            NULL AS {C.YEAR_COL}, 
            val AS {C.VAL_COL}
        FROM unpivoted
    )
    SELECT * FROM result_rel_elev
    """
    result_rel_elev = conn.execute(query).df()
    # clear temporary table
    conn.execute("DROP INDEX IF EXISTS rtree_aoi_elevation")
    conn.execute("DROP TABLE IF EXISTS chunk")
    conn.execute("DROP TABLE IF EXISTS aoi_elevation")
    conn.unregister('chunk_wkt')
    conn.unregister('buffer_size')
    conn.unregister('ref_elevation')
    # done
    result = pd.concat([result_rel_elev, result_ref_elev])
    return result


@typechecked
def _worker(task_queue: mp.Queue,
            result_queue: mp.Queue,
            table: str,
            buffer_sizes: list[float], 
            table_path: str | Path,
            memory_limit: str = "4GB"):
    """worker loop"""
    conn = generate_duckdb_memory_connection(memory_limit=memory_limit)
    while True:
        try:
            task = task_queue.get(timeout=0.1)
            if isinstance(task, str):
                if task == C.SENTINEL:
                    result_queue.put(C.SENTINEL)
                    break
            chunk = task  # pandas DataFrame [ID_COL, wkt]
            res = _query(chunk, table, buffer_sizes, table_path, conn)
            result_queue.put((len(chunk), res))
        except queue.Empty:
            continue
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
        is_valid_elevation = all(et in VALID_ELEVATION_TYPES for et in elevation_types)
        if not is_valid_elevation:
            raise ValueError(f"Invalid elevation type. Valid types are: {VALID_ELEVATION_TYPES}")
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
            description = TQDM_DESC(table, buffer_sizes)
            tq = tqdm(total=len(self.wkt_df), bar_format=C.TQDM_BAR_FORMAT, desc=description, disable=not self.verbose)
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
