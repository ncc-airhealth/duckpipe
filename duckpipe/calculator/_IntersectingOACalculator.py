import pandas as pd
from typeguard import typechecked
from typing import Self, Tuple
from duckpipe.calculator.Worker import _worker
from tqdm import tqdm
from duckpipe.common import TQDM_BAR_FORMAT

TABLE_NAME = 'jgg_borders_2023'


@typechecked
def _normalize_params(buffer_sizes: float | list[float]) -> list[float]:
    if isinstance(buffer_sizes, float):
        buffer_sizes = [buffer_sizes]
    buffer_sizes = sorted(buffer_sizes)
    return buffer_sizes

@typechecked
def _generate_query(buffer_sizes, table_path: str) -> Tuple[str, str, str]:
    """
    Generate DuckDB queries for intersecting OA ratio.
    create 'oa_intersection_ratio' temp table with columns:
    - id: int — Chunk ID.
    - buffer_size: float — Buffer size.
    - tot_reg_cd: int — OA code.
    - intersection_ratio: float — Intersection ratio of OA and buffered chunk.

    [input]
    - buffer_sizes: list[float] — Buffer sizes.
    - table_path: str — Path to OA table.

    [output]
    - Tuple[str, str, str] — Pre-query, main query, and post-query.
    """
    # buffer size values clause
    values_clause = ", ".join(f"({bs})" for bs in buffer_sizes)
    max_buffer_size = max(buffer_sizes)
    pre_query = '\n'.join([
        f"""
        CREATE OR REPLACE TEMP TABLE buffer_size AS (
            SELECT * FROM (VALUES {values_clause}) AS t(buffer_size)
        );
        """,
        f"""
        CREATE OR REPLACE TEMP TABLE aoi_oa AS (
            WITH 
            aoi AS (
                SELECT
                    MIN(ST_XMin(geometry)) - {max_buffer_size} AS xmin,
                    MIN(ST_YMin(geometry)) - {max_buffer_size} AS ymin,
                    MAX(ST_XMax(geometry)) + {max_buffer_size} AS xmax,
                    MAX(ST_YMax(geometry)) + {max_buffer_size} AS ymax, 
                    ST_MakeEnvelope(
                        MIN(ST_XMin(geometry)) - {max_buffer_size}, 
                        MIN(ST_YMin(geometry)) - {max_buffer_size}, 
                        MAX(ST_XMax(geometry)) + {max_buffer_size}, 
                        MAX(ST_YMax(geometry)) + {max_buffer_size}
                    ) AS aoi
                FROM chunk
            ),
            filtered AS (
                SELECT 
                    t.tot_reg_cd, 
                    t.geometry AS geometry
                FROM '{table_path}' AS t
                INNER JOIN aoi AS a ON
                    t.xmin < a.xmax AND 
                    t.xmax > a.xmin AND 
                    t.ymin < a.ymax AND 
                    t.ymax > a.ymin
            )
            SELECT * 
            FROM filtered
        );
        CREATE INDEX rtree_aoi_oa
        ON aoi_oa 
        USING RTREE (geometry) WITH (max_node_capacity = 4);
        """, 
        f"""
        CREATE OR REPLACE TEMP TABLE oa_intersection_ratio AS (
            WITH 
            chunk_buffer AS (
                SELECT c.id, bs.buffer_size, ST_Buffer(c.geometry, bs.buffer_size) AS geometry 
                FROM chunk AS c
                CROSS JOIN buffer_size AS bs
            ), 
            chunk_buffer_oa AS (
                SELECT 
                    cb.id, 
                    cb.buffer_size, 
                    ao.tot_reg_cd, 
                    COALESCE(ST_Area(ST_Intersection(cb.geometry, ao.geometry)) / ST_Area(ao.geometry), 0) AS intersection_ratio
                FROM 
                    chunk_buffer AS cb
                LEFT JOIN 
                    aoi_oa AS ao ON ST_Intersects(ao.geometry, cb.geometry)
                ORDER BY
                    id, tot_reg_cd
            )
            SELECT * FROM chunk_buffer_oa
        );
        """
    ])
    main_query = "SELECT * FROM oa_intersection_ratio;"
    post_query = """
        DROP INDEX IF EXISTS rtree_aoi_oa;
        DROP TABLE IF EXISTS aoi_oa;
        DROP TABLE IF EXISTS buffer_size;
        DROP TABLE IF EXISTS oa_intersection_ratio;
    """
    return pre_query, main_query, post_query


class IntersectingOACalculator:

    @typechecked
    def _calculate_intersecting_oa(self, buffer_sizes: float | list[float]) -> Self:
        # normalize input
        buffer_sizes = _normalize_params(buffer_sizes)
        # prepare
        self.result_df_temp = self.result_df
        self.result_df = pd.DataFrame()
        # query
        table_path = f"{self.data_dir}/{TABLE_NAME}.parquet"
        desc = f"caching intersecting jgg ratio ({buffer_sizes})"
        pre_query, main_query, post_query = _generate_query(buffer_sizes, table_path)
        self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        self.oa_intersection_df = self.result_df
        self.result_df = self.result_df_temp
        return self