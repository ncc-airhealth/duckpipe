"""
[description]
Airport distance calculator. Computes minimum distance from each feature to
airports for requested years using DuckDB Spatial over geometry chunks.
"""

from typeguard import typechecked
from typing import Self, Tuple
from duckpipe.calculator._IntersectingOACalculator import _generate_query as oa_ratio_query

VALID_YEARS = [2000, 2005, 2010, 2015, 2020]
TABLE_NAME = "jgg_adjusted_sgis_ho_yr"
OA_TABLE_NAME = "jgg_borders_2023"
VAR_NAME_MACRO = """
CREATE OR REPLACE MACRO varname(ho_yr_code, buffer_size) AS (
    printf('%s_%d_%04d', 'ho_yr', ho_yr_code, buffer_size::INTEGER)
);
"""


@typechecked
def _normalize_params(years: int | list[int], buffer_sizes: float | list[float]) -> Tuple[list[int], list[float]]:
    # normalize input type
    if isinstance(years, int):
        years = [years]
    if isinstance(buffer_sizes, float):
        buffer_sizes = [buffer_sizes]
    # sort
    years = sorted(years)
    buffer_sizes = sorted(buffer_sizes)
    # check
    for year in years:
        if year not in VALID_YEARS:
            msg = f"Invalid year '{year}'. Valid years are: {VALID_YEARS}"
            raise ValueError(msg)
    # return
    return years, buffer_sizes

@typechecked
def _generate_query(year: int, buffer_sizes: list[float], table_path: str, oa_table_path: str) -> Tuple[str, str, str]:
    oa_pre_query, oa_main_query, oa_post_query = oa_ratio_query(buffer_sizes, oa_table_path)
    pre_query = '\n'.join([
        oa_pre_query, 
        VAR_NAME_MACRO
    ])
    main_query = '\n'.join([
        oa_main_query, 
        f"""
        WITH
        ho_yr_codes AS (
            SELECT DISTINCT ho_yr_code FROM '{table_path}'
        ), 
        chunk_ids AS (
            SELECT DISTINCT id FROM chunk
        ), 
        result_skeleton AS (
            SELECT 
                ci.id, 
                hc.ho_yr_code,
                
            FROM ho_yr_codes AS hc
            CROSS JOIN chunk_ids AS ci
            
        )
        ho_yr_sel AS (
            SELECT tot_reg_cd, year, ho_yr_code, value
            FROM '{table_path}'
            WHERE year = {year}
        ), 
        agg AS (
            SELECT 
                oa.id, 
                varname(t.ho_yr_code, oa.buffer_size) AS varname,
                t.year AS year, 
                SUM(COALESCE(t.value * oa.intersection_ratio, 0)) AS value
            FROM 
                oa_intersection_ratio AS oa
            LEFT JOIN ho_yr_sel AS t
            ON (oa.tot_reg_cd = t.tot_reg_cd)
            GROUP BY 
                oa.id, 
                t.year,
                oa.buffer_size, 
                t.ho_yr_code
        )
        SELECT * FROM agg;
        """
    ])
    post_query = '\n'.join([
        oa_post_query,
    ])
    return pre_query, main_query, post_query


class HouseYearCalculator:

    @typechecked
    def calculate_house_year(self, years: int | list[int], buffer_sizes: float | list[float]) -> Self:
        """
        Calculate per-feature house year for one or more years.
        """
        # normalize input
        years, buffer_sizes = _normalize_params(years, buffer_sizes)
        # generate requests per year
        table_path = f"{self.data_dir}/{TABLE_NAME}.parquet"
        oa_table_path = f"{self.data_dir}/{OA_TABLE_NAME}.parquet"
        for year in years:
            pre_query, main_query, post_query = _generate_query(year, buffer_sizes, table_path, oa_table_path)
            desc = f"House year ({year})"
            self.run_query_workers(pre_query, main_query, post_query, mode=self.worker_mode, desc=desc)
        # done
        return self