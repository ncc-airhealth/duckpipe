"""
[description]
Common constants and settings for duckpipe (CRS, schema/names, std columns,
sentinels, tqdm formatting).
"""
REF_EPSG = 5179
DB_NAME = "main"

YEAR_COL = "year"
VAR_COL = "varname"
VAL_COL = "value"
ID_COL = "id"

SENTINEL = "__END__"

TQDM_BAR_FORMAT = "| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]  {desc}"
