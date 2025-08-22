"""
Common constants and settings shared across the duckpipe package.

This module defines reference CRS, database/schema names, standardized column names
for outputs, multiprocessing sentinels, and tqdm display settings used by calculators.
"""
REF_EPSG = 5179
DB_NAME = "main"

YEAR_COL = "year"
VAR_COL = "stat_alias"
VAL_COL = "value"
ID_COL = "uuid_35ab93c72f484478a4cab4233aa3d434"

SENTINEL = "__END__"

TQDM_BAR_FORMAT = "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]  {desc}"
