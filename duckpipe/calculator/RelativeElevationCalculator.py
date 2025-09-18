"""
[description]
Relative elevation calculator. Computes donut-based relative elevation ratios and
reference elevation per feature using DuckDB Spatial.
"""
from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path


DONUT_THICKNESS = 30
DEM_SPATIAL_RESOLUTION = 30
VALID_ELEVATION_TYPES = ["dem", "dsm"]
VAR_NAME_MACRO_REL = """
CREATE OR REPLACE MACRO varname_rel(elev_type, stat, buffer_size) AS (
    printf('%s_%s_%s', 
        CASE
            WHEN elev_type = 'dem' THEN 'Alt_k'
            WHEN elev_type = 'dsm' THEN 'Alt_a'
            ELSE 'error_processing_relative_elevation'
        END, 
        stat, 
        buffer_size::VARCHAR
    )
);
"""
VAR_NAME_MACRO_REF = """
CREATE OR REPLACE MACRO varname_ref(elev_type) AS (
    CASE
        WHEN elev_type = 'dem' THEN 'Altitude_k'
        WHEN elev_type = 'dsm' THEN 'Altitude_a'
        ELSE 'error_processing_relative_elevation'
    END
);
"""


@typechecked
def _normalize_params(elevation_types: str | list[str], buffer_sizes: float | list[float]) -> Tuple[list[str], list[float]]:
    if isinstance(elevation_types, str):
        elevation_types = [elevation_types]
    if isinstance(buffer_sizes, float):
        buffer_sizes = [buffer_sizes]
    elevation_types = sorted(elevation_types)
    buffer_sizes = sorted(buffer_sizes)
    for et in elevation_types:
        if et not in VALID_ELEVATION_TYPES:
            raise ValueError(f"Invalid elevation type '{et}'. Valid types are: {VALID_ELEVATION_TYPES}")
    return elevation_types, buffer_sizes


def _generate_query(elevation_type: str, table_path: Path, buffer_sizes: list[float]) -> tuple[str, str, str]:
    values_clause = ", ".join(f"({bs})" for bs in buffer_sizes)
    max_buffer_size = max(buffer_sizes)
    clip_distance = max_buffer_size + DONUT_THICKNESS + 2 * DEM_SPATIAL_RESOLUTION
    pre_query = "\n".join([
        VAR_NAME_MACRO_REL,
        VAR_NAME_MACRO_REF,
        f"""
        CREATE OR REPLACE TEMP TABLE buffer_size AS (
            SELECT * FROM (VALUES {values_clause}) AS t(buffer_size)
        );
        """,
        f"""
        CREATE OR REPLACE TEMP TABLE aoi_elevation AS (
            WITH 
            aoi AS (
                SELECT
                    MIN(ST_XMin(geometry)) - {clip_distance} AS xmin,
                    MIN(ST_YMin(geometry)) - {clip_distance} AS ymin,
                    MAX(ST_XMax(geometry)) + {clip_distance} AS xmax,
                    MAX(ST_YMax(geometry)) + {clip_distance} AS ymax
                FROM chunk
            ),
            filtered AS (
                SELECT 
                    ST_MakeEnvelope(t.xmin, t.ymin, t.xmax, t.ymax) AS geometry,
                    COALESCE(t.value, 0) AS elev
                FROM '{table_path}' AS t
                INNER JOIN aoi AS a
                ON t.xmin > a.xmin AND t.xmax < a.xmax AND t.ymin > a.ymin AND t.ymax < a.ymax
            )
            SELECT elev, geometry
            FROM filtered
            WHERE NOT ST_IsEmpty(geometry)
        );
        CREATE INDEX rtree_aoi_elevation ON aoi_elevation USING RTREE (geometry) WITH (max_node_capacity = 4);
        """,
        f"""
        CREATE OR REPLACE TEMP TABLE ref_elevation AS (
            SELECT 
                id, 
                MEAN(a.elev) AS ref_elev
            FROM chunk AS c
            LEFT JOIN aoi_elevation AS a ON ST_Intersects(c.geometry, a.geometry)
            GROUP BY id
        );
        """,
    ])
    main_query = f"""
        WITH 
        donut AS (
            SELECT 
                c.id, 
                bs.buffer_size, 
                re.ref_elev, 
                ST_Difference(
                    ST_Buffer(c.geometry, bs.buffer_size + {DONUT_THICKNESS}), 
                    ST_Buffer(c.geometry, bs.buffer_size)
                ) AS geometry
            FROM chunk AS c
            CROSS JOIN buffer_size AS bs
            INNER JOIN ref_elevation AS re ON c.id = re.id
        )
        , rel_elevation_ratio AS (
            SELECT 
                d.id, 
                d.buffer_size, 
                AVG(CAST((a.elev - d.ref_elev) > +20 AS INT)) AS above_20,
                AVG(CAST((a.elev - d.ref_elev) < -20 AS INT)) AS below_20,
                AVG(CAST((a.elev - d.ref_elev) > +50 AS INT)) AS above_50,
                AVG(CAST((a.elev - d.ref_elev) < -50 AS INT)) AS below_50
            FROM aoi_elevation AS a
            INNER JOIN donut AS d ON ST_Intersects(d.geometry, a.geometry)
            GROUP BY d.id, d.buffer_size
        )
        , unpivoted AS (
            SELECT *
            FROM rel_elevation_ratio
            UNPIVOT ( val FOR stat IN (above_20, below_20, above_50, below_50) )
        )
        , result_rel_elev AS (
            SELECT 
                id, 
                varname_rel('{elevation_type}', stat, buffer_size) AS varname, 
                NULL AS year, 
                val AS value
            FROM unpivoted
        )
        , result_ref AS (
            SELECT 
                id, 
                varname_ref('{elevation_type}') AS varname, 
                NULL AS year, 
                ref_elev AS value
            FROM ref_elevation
        )
        SELECT * FROM result_rel_elev
        UNION ALL
        SELECT * FROM result_ref
    """
    post_query = """
        DROP INDEX IF EXISTS rtree_aoi_elevation;
        DROP TABLE IF EXISTS aoi_elevation;
        DROP TABLE IF EXISTS ref_elevation;
        DROP TABLE IF EXISTS buffer_size;
    """
    return pre_query, main_query, post_query


class RelativeElevationCalculator:

    @typechecked
    def calculate_relative_elevation(self, 
                                     elev_types: str | list[str], 
                                     buffer_sizes: float | list[float]
                                     ) -> Self:
        """
        [description]
        Calculate relative elevation metrics for one or more elevation types and
        buffer sizes using the standardized worker runner (`run_query_workers`).

        [input]
        - elev_types: str | list[str] — Elevation source(s) to compute (in `VALID_ELEVATION_TYPES`).
        - buffer_sizes: float | list[float] — Buffer distances for donut rings.

        [output]
        - Self — Appends rows with [`id`, `varname`, `year` (NULL), `value`] and returns self.
        """
        elev_types, buffer_sizes = _normalize_params(elev_types, buffer_sizes)
        for elev_type in elev_types:
            table_path = self.data_dir / f"{elev_type}.parquet"
            pre_query, main_query, post_query = _generate_query(elev_type, table_path, buffer_sizes)
            desc = f"Relative elevation ({elev_type}) (buffer_sizes: {buffer_sizes})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        return self
