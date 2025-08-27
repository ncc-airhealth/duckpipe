import duckdb
from duckdb import DuckDBPyConnection
from typeguard import typechecked
from duckpipe.common import *


DUCKDB_EXTENSIONS = ["spatial", "httpfs"]

@typechecked
def install_duckdb_extensions():
    """
    [description]
    Install and load required DuckDB extensions used across the project. Currently installs
    and loads: `spatial` and `httpfs`.

    [input]
    - None

    [output]
    - None — Side effect: installs and loads extensions into a temporary connection.

    [example usage]
    ```python
    from duckpipe.duckdb_utils import install_duckdb_extensions
    install_duckdb_extensions()
    ```
    """
    conn = duckdb.connect()
    for ext in DUCKDB_EXTENSIONS:
        conn.execute(f"INSTALL {ext}")
        conn.execute(f"LOAD {ext}")
    conn.close()
    return

@typechecked
def generate_duckdb_connection(db_path: str, memory_limit: str="6GB") -> DuckDBPyConnection:
    """
    [description]
    Create a new DuckDB connection, configure it for geospatial workloads, attach the database
    at `db_path` as read-only, and set the active schema to `DB_NAME` from `duckpipe.common`.

    [input]
    - db_path: str — Path to DuckDB database file to attach (read-only).
    - memory_limit: str — Memory limit string (e.g., "6GB").

    [output]
    - duckdb.DuckDBPyConnection — Ready-to-use connection with Spatial loaded.

    [example usage]
    ```python
    from duckpipe.duckdb_utils import install_duckdb_extensions, generate_duckdb_connection
    install_duckdb_extensions()
    conn = generate_duckdb_connection("example.duckdb", memory_limit="6GB")
    # ... use conn ...
    conn.close()
    ```
    """
    conn = duckdb.connect(database=db_path, read_only=True)
    conn.execute("PRAGMA threads=1") # geospatial processing better when single threaded
    conn.execute("LOAD spatial")
    conn.execute(f"SET memory_limit='{memory_limit}'")
    conn.execute("SET enable_progress_bar = false")
    return conn