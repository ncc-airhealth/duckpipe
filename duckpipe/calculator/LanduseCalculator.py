"""
[description]
Land-use calculator that computes area and proportion statistics by land-use code
for requested years and buffer sizes using DuckDB Spatial over geometry chunks.
"""

from typeguard import typechecked
from typing import Self, Tuple
from pathlib import Path

VALID_YEARS = [2000, 2005, 2010, 2015, 2020]
VAR_PREFIX = "LS"
VAR_NAME_MACRO = """
CREATE OR REPLACE MACRO varname(lu_code, buffer_size, stat_type) AS (
    -- stat_type: a (area) or p (proportion)
    printf('%s%s_%04d_%s', 'LS', lu_code, buffer_size::INTEGER, stat_type)
);
"""


@typechecked
def _normalize_params(years: int | list[int], buffer_sizes: float | list[float] | None) -> Tuple[list[int], list[float]]:
    # normalize
    if isinstance(years, int):
        years = [years]
    if buffer_sizes is None:
        buffer_sizes = [0]
    elif isinstance(buffer_sizes, float):
        buffer_sizes = [buffer_sizes]
    # sort
    years = sorted(years)
    buffer_sizes = sorted(buffer_sizes)
    # validate
    for year in years:
        if year not in VALID_YEARS:
            raise ValueError(f"Invalid year '{year}'. Valid years are: {VALID_YEARS}")
    return years, buffer_sizes

def _generate_query(year: int, table_path: Path, buffer_sizes: list[float]) -> Tuple[str, str, str]:
    # buffer size values clause
    values_clause = ", ".join(f"({bs})" for bs in buffer_sizes)
    max_buffer_size = max(buffer_sizes)
    pre_query = '\n'.join([
        VAR_NAME_MACRO,
        f"""
        CREATE OR REPLACE TEMP TABLE buffer_size AS (
            SELECT * 
            FROM (VALUES {values_clause}) AS t(buffer_size)
        );
        """,
        f"""
        CREATE OR REPLACE TEMP TABLE result_skeleton AS (
            WITH codes AS (
                SELECT DISTINCT code FROM '{table_path}'
                WHERE year = {year}
            )
            SELECT DISTINCT c.id, t.code, bs.buffer_size
            FROM codes AS t, chunk AS c, buffer_size AS bs
            ORDER BY c.id, t.code, bs.buffer_size
        );
        """, 
        f"""
        CREATE OR REPLACE TEMP TABLE aoi_landuse AS (
            WITH 
            aoi AS (
                SELECT
                    MIN(ST_XMin(geometry)) - {max_buffer_size} AS xmin,
                    MIN(ST_YMin(geometry)) - {max_buffer_size} AS ymin,
                    MAX(ST_XMax(geometry)) + {max_buffer_size} AS xmax,
                    MAX(ST_YMax(geometry)) + {max_buffer_size} AS ymax
                FROM chunk
            ),
            filtered AS (
                SELECT 
                    code, 
                    ST_Intersection(
                        t.geometry, 
                        ST_MakeEnvelope(a.xmin, a.ymin, a.xmax, a.ymax)
                    ) AS geometry
                FROM aoi AS a
                INNER JOIN '{table_path}' AS t ON
                    t.xmin < a.xmax AND 
                    t.xmax > a.xmin AND 
                    t.ymin < a.ymax AND 
                    t.ymax > a.ymin
                WHERE t.year = {year}
            )
            SELECT * FROM filtered WHERE NOT ST_IsEmpty(geometry)
        );
        CREATE INDEX rtree_aoi_landuse
        ON aoi_landuse 
        USING RTREE (geometry) WITH (max_node_capacity = 4);
        """, 
    ])
    main_query = f"""
        WITH 
        aoi AS (
            SELECT 
                c.id, 
                b.buffer_size, 
                ST_Area(ST_Buffer(c.geometry, b.buffer_size)) AS area, 
                ST_Buffer(c.geometry, b.buffer_size) AS geometry
            FROM 
                chunk AS c, 
                buffer_size AS b
        ),
        aggregated AS (
            SELECT
                a.id, 
                a.buffer_size, 
                CAST(l.code AS VARCHAR) AS lu_code, 
                SUM( ST_Area(ST_Intersection(l.geometry, a.geometry)) ) AS a, 
                SUM( ST_Area(ST_Intersection(l.geometry, a.geometry)) / a.area ) AS p
            FROM 
                aoi_landuse AS l
            INNER JOIN 
                aoi AS a ON ST_Intersects(l.geometry, a.geometry)
            GROUP BY 
                a.id, 
                l.code, 
                a.buffer_size
        ),
        aggregated_filled AS (
            SELECT
                rs.id, 
                rs.buffer_size, 
                rs.code::VARCHAR AS lu_code,
                COALESCE(a.a, 0) AS a, 
                COALESCE(a.p, 0) AS p
            FROM 
                aggregated AS a
            RIGHT JOIN 
                result_skeleton AS rs 
            ON 
                rs.id = a.id AND
                rs.code = a.lu_code AND
                rs.buffer_size = a.buffer_size
        ),
        unpivoted AS (
            SELECT * FROM 
            aggregated_filled 
            UNPIVOT ( val FOR stat_type IN (a, p) )
        ),
        renamed AS (
            SELECT 
                id, 
                varname(lu_code, buffer_size, stat_type) AS varname, 
                {year} AS year, 
                val AS value
            FROM unpivoted
        )
        SELECT * FROM renamed ORDER BY id, varname;
    """
    post_query = """
        DROP INDEX IF EXISTS rtree_aoi_landuse;
        DROP TABLE IF EXISTS aoi_landuse;
        DROP TABLE IF EXISTS result_skeleton;
        DROP TABLE IF EXISTS buffer_size;
    """
    return pre_query, main_query, post_query


class LanduseCalculator:

    @typechecked
    def calculate_landuse_area_ratio(self,
                                     years: int | list[int], 
                                     buffer_sizes: float | list[float] | None
                                     ) -> Self:
        """
        [description]
        Calculate land-use `area` and `ratio` statistics for one or more `years` and one or more
        `buffer_sizes` using the standardized worker runner (`run_query_workers`).

        [input]
        - years: int | list[int] — One or more years (must be in `VALID_YEARS`).
        - buffer_sizes: float | list[float] | None — Buffer distances; if None, defaults to `[0]`.

        [output]
        - Self — Returns self for chaining. Appends rows with [`C.ID_COL`, `C.VAR_COL`,
          `C.YEAR_COL`, `C.VAL_COL`].
        """
        # normalize params
        years, buffer_sizes = _normalize_params(years, buffer_sizes)
        # run per year
        for year in years:
            table_path = (self.data_dir / f"landuse_{year}").with_suffix(".parquet")
            pre_query, main_query, post_query = _generate_query(year, table_path, buffer_sizes)
            desc = f"Landuse ({year}) (buffer_sizes: {buffer_sizes})"
            self.run_query_workers(pre_query, main_query, post_query, desc=desc)
        return self