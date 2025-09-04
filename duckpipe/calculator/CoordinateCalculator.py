import pandas as pd
from typeguard import typechecked
from typing import Self
from tqdm import tqdm

import duckpipe.common as C


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

class CoordinateCalculator:

    @typechecked
    def calculate_coordinate(self, mode: str="centroid") -> Self:
        # input check
        _supported_modes = [k for k in SUPPORTED_MODE_FUNCS.keys()]
        if mode not in _supported_modes:
            raise ValueError(f"Invalid mode. Valid modes are: {_supported_modes}")
        # perform calculations
        coord_func = SUPPORTED_MODE_FUNCS[mode]
        self.conn.register('full_wkt', self.geom_df)
        query = f"""
        WITH 
        raw_coords AS (
            SELECT 
                {C.ID_COL}, 
                ST_Point(
                    ST_Y({coord_func}(ST_GeomFromText(wkt))), -- EPSG 5179 X/Y flipped
                    ST_X({coord_func}(ST_GeomFromText(wkt)))  -- EPSG 5179 X/Y flipped
                ) AS geometry
            FROM full_wkt
        ),
        gcs_xcoords AS (
            SELECT 
                {C.ID_COL}, 
                '{GCS_VAR_X}' AS {C.VAR_COL}, 
                ST_Y( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{GCS_EPSG}') ) AS {C.VAL_COL}
            FROM raw_coords
        ),
        gcs_ycoords AS (
            SELECT 
                {C.ID_COL}, 
                '{GCS_VAR_Y}' AS {C.VAR_COL}, 
                ST_X( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{GCS_EPSG}') ) AS {C.VAL_COL}
            FROM raw_coords
        ),
        pcs_xcoords AS (
            SELECT 
                {C.ID_COL}, 
                '{PCS_VAR_X}' AS {C.VAR_COL}, 
                ST_Y( ST_Transform(geometry, 'EPSG:{C.REF_EPSG}', 'EPSG:{PCS_EPSG}') ) AS {C.VAL_COL}
            FROM raw_coords
        ), 
        pcs_ycoords AS (
            SELECT 
                {C.ID_COL}, 
                '{PCS_VAR_Y}' AS {C.VAR_COL}, 
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
        df = self.conn.execute(query).df()
        self.conn.unregister('full_wkt')
        # done.
        self.result_df = pd.concat([self.result_df, df], ignore_index=True)
        return self