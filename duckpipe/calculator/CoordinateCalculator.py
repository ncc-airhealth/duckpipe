"""
Coordinate calculator to derive representative coordinates (centroid or point-on-surface)
for each input geometry and emit both projected (TM) and geographic (WGS84) coordinates.
"""

from typeguard import typechecked
from typing import Self, Tuple
from duckpipe.common import REF_EPSG


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


@typechecked
def _normalize_params(mode: str) -> str:
    modes = list(SUPPORTED_MODE_FUNCS.keys())
    if mode not in modes:
        raise ValueError(f"Invalid mode. Valid modes are: {modes}")
    return mode


@typechecked
def _generate_query(mode: str) -> Tuple[str, str, str]:
    coord_func = SUPPORTED_MODE_FUNCS[mode]
    pre_query = ""
    main_query = f"""
        WITH 
        geom_tbl AS (
            SELECT 
                id, 
                {coord_func}(geometry) AS centroid
            FROM chunk
        ), 
        coord_tbl AS (
            SELECT 
                id, 
                NULL AS year, 
                ST_X(centroid) AS {PCS_VAR_X}, 
                ST_Y(centroid) AS {PCS_VAR_Y}, 
                ST_X(ST_Transform(
                    centroid, 
                    'EPSG:{REF_EPSG}', 
                    'EPSG:{GCS_EPSG}', 
                    always_xy := true
                )) AS {GCS_VAR_X}, 
                ST_Y(ST_Transform(
                    centroid, 
                    'EPSG:{REF_EPSG}', 
                    'EPSG:{GCS_EPSG}', 
                    always_xy := true
                )) AS {GCS_VAR_Y}
            FROM geom_tbl
        ), 
        result AS (
            SELECT *
            FROM coord_tbl
            UNPIVOT ( value FOR varname IN ({GCS_VAR_X}, {GCS_VAR_Y}, {PCS_VAR_X}, {PCS_VAR_Y}) )
        )
        SELECT * FROM result
    """
    post_query = ""
    return pre_query, main_query, post_query


class CoordinateCalculator:

    @typechecked
    def calculate_coordinate(self, mode: str="centroid") -> Self:
        """
        Compute representative coordinates for all features using the standardized
        worker runner over precomputed `self.chunks`, and append results to `self.result_df`.

        [input]
        - mode: str — One of {"centroid", "representative_point"}. Defaults to "centroid".

        [output]
        - Self — Returns self for chaining.

        [example usage]
        ```python
        calculator = calculator.calculate_coordinate(mode="centroid")
        ```
        """
        mode = _normalize_params(mode)
        pre_query, main_query, post_query = _generate_query(mode)
        desc = f"Coordinate ({mode})"
        self.run_query_workers(pre_query, main_query, post_query, desc=desc)
        return self