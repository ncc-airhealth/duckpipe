"""
[description]
Chunking utilities to split input geometries into spatially coherent chunks for
efficient memory locality and parallel processing.
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typeguard import typechecked
from typing import Self
from tqdm import tqdm

import duckpipe.common as C

class Clustering:
    """
    [description]
    Utilities to split the input geometries into spatially coherent chunks to improve
    memory locality and parallel processing efficiency.

    [example usage]
    ```python
    # from test/example.py
    calculator = dp.Calculator(db_path=DB_PATH, n_workers=2, memory_limit="6GB")
    calculator = (
        calculator
        .set_dataframe(gdf)
        .chunk_by_centroid(max_cluster_size=MAX_CLUSTER_SIZE, distance_threshold=MAX_CLUSTER_WIDTH)
    )
    chunks = calculator.get_chunks()
    ```
    """
    
    @typechecked
    def get_chunks(self):
        """
        [description]
        Return the list of chunks prepared by a prior call to a chunking method.

        [input]
        - None

        [output]
        - list[pandas.DataFrame] — Each chunk is a DataFrame with columns [`C.ID_COL`, "wkt"].

        [example usage]
        ```python
        chunks = calculator.get_chunks()
        ```
        """
        return self.chunks

    @typechecked
    def chunk_by_centroid(self, max_cluster_size: int=50, distance_threshold: float=2000, max_rows: int=20000) -> Self:
        """
        [description]
        Hierarchical clustering on point centroids (complete-linkage, Euclidean) with a maximum
        distance threshold. Large clusters are further split into fixed-size slices.

        [input]
        - max_cluster_size: int — Maximum rows per chunk after slicing.
        - distance_threshold: float — Maximum centroid distance within a cluster (CRS units).

        [output]
        - Self — The same Calculator instance with `self.chunks` populated.

        [example usage]
        ```python
        calculator = calculator.chunk_by_centroid(max_cluster_size=100, distance_threshold=10000)
        ```
        """
        # call geometry
        nrows = len(self.wkt_df)
        if nrows == 0:
            self.chunks = []
            return self
        elif nrows == 1:
            self.chunks = [self.wkt_df[[C.ID_COL, "wkt"]]]
            return self
        # preprocessing

        self.conn.register("wkt_tbl", self.wkt_df)
        query = f"""
        SELECT 
            {C.ID_COL}, 
            ST_X(ST_Centroid(ST_GeomFromText(wkt))) AS x,
            ST_Y(ST_Centroid(ST_GeomFromText(wkt))) AS y
        FROM wkt_tbl
        """
        centroid_df = self.conn.execute(query).df()
        self.conn.unregister('wkt_tbl')
        # clustering
        self.chunks = []
        tq = tqdm(total=nrows, bar_format=C.TQDM_BAR_FORMAT, desc="chunking", disable=not self.verbose)
        for idx0 in range(0, nrows, max_rows):
            # dividing for memory efficiency 
            idx1 = min(idx0 + max_rows, nrows)
            id_sr = self.wkt_df.iloc[idx0:idx1][C.ID_COL]
            wkt_sr = self.wkt_df.iloc[idx0:idx1]["wkt"]
            x_arr = centroid_df.iloc[idx0:idx1]["x"].to_numpy()
            y_arr = centroid_df.iloc[idx0:idx1]["y"].to_numpy()
            # feature
            X = np.column_stack([x_arr, y_arr])
            Z = linkage(X, method="complete", metric="euclidean")
            labels = fcluster(Z, t=distance_threshold, criterion="distance") - 1
            cdf = pd.DataFrame({
                C.ID_COL: id_sr,
                "wkt": wkt_sr,
                "cluster": labels,
            })
            # chunking
            for _, chunk in cdf.groupby("cluster", sort=False):
                if len(chunk) <= max_cluster_size:
                    self.chunks.append(chunk[[C.ID_COL, "wkt"]])
                else:
                    new_chunks = [
                        chunk.iloc[i:i + max_cluster_size]
                        for i in range(0, len(chunk), max_cluster_size)
                    ]
                    self.chunks.extend(new_chunks)
            tq.update(idx1 - idx0)
        tq.close()
        # end
        self.chunks = sorted(self.chunks, key=len, reverse=True)
        return self
    
    @typechecked
    def chunk_by_order(self, max_cluster_size: int=50) -> Self:
        """
        [description]
        Order features by their original order and slice into fixed-size chunks.

        [input]
        - max_cluster_size: int — Maximum rows per chunk.

        [output]
        - Self — The same Calculator instance with `self.chunks` populated.

        [example usage]
        ```python
        calculator = calculator.chunk_by_order(max_cluster_size=100)
        ```
        """
        # call geometry
        cdf = self.wkt_df.loc[:, [C.ID_COL, "wkt"]]
        # clustering
        chunks = [
            cdf.iloc[i:i + max_cluster_size]
            for i in range(0, len(cdf), max_cluster_size)
        ]
        self.chunks = chunks
        # end
        return self
    
    @typechecked
    def chunk_by_hilbert(self, max_cluster_size: int=50) -> Self:
        """
        [description]
        Order features by Hilbert curve distance and slice into fixed-size chunks.

        [input]
        - max_cluster_size: int — Maximum rows per chunk.

        [output]
        - Self — The same Calculator instance with `self.chunks` populated.

        [example usage]
        ```python
        calculator = calculator.chunk_by_hilbert(max_cluster_size=100)
        ```
        """
        # call geometry
        self.conn.register("wkt_df", self.wkt_df)
        query = """
        SELECT *
        FROM wkt_df
        ORDER BY ST_Hilbert(ST_GeomFromText(wkt))
        """
        hilbert_df = self.conn.execute(query).df()
        self.conn.unregister('wkt_df')
        # clustering
        chunks = [
            hilbert_df.iloc[i:i + max_cluster_size]
            for i in range(0, len(hilbert_df), max_cluster_size)
        ]
        self.chunks = chunks
        # end
        return self
        




    
