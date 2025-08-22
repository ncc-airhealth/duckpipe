import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from typeguard import typechecked
from typing import Self

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
    def chunk_by_centroid(self, max_cluster_size: int=50, distance_threshold: float=2000) -> Self:
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
        cdf = self.geom_df.loc[:, [C.ID_COL, "wkt"]].copy()
        geom_sr = gpd.GeoSeries.from_wkt(cdf["wkt"]).set_crs(C.REF_EPSG, allow_override=True)
        centroid = geom_sr.centroid
        # feature matrix
        X = np.column_stack([centroid.x.to_numpy(), centroid.y.to_numpy()])
        # clustering
        if len(X) == 0:
            self.chunks = []
            return self
        if len(X) == 1:
            cdf["cluster"] = 0
        else:
            Z = linkage(X, method="complete", metric="euclidean")
            labels = fcluster(Z, t=distance_threshold, criterion="distance") - 1
            cdf["cluster"] = labels
        chunks = [
            chunk[[C.ID_COL, "wkt"]]
            for _, chunk in cdf.groupby("cluster", sort=False)
        ]
        # max cluster size check
        self.chunks = []
        for chunk in chunks:
            if len(chunk) <= max_cluster_size:
                self.chunks.append(chunk)
            else:
                new_chunks = [
                    chunk.iloc[i:i + max_cluster_size]
                    for i in range(0, len(chunk), max_cluster_size)
                ]
                self.chunks.extend(new_chunks)
        self.chunks = sorted(self.chunks, key=len, reverse=True)
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
        cdf = self.geom_df.loc[:, [C.ID_COL, "wkt"]]
        # clustering
        geom_sr = gpd.GeoSeries.from_wkt(cdf["wkt"]).set_crs(C.REF_EPSG, allow_override=True)
        centroid = geom_sr.centroid
        cdf["hilbert_distance"] = centroid.hilbert_distance()
        cdf.sort_values(by="hilbert_distance", inplace=True)
        cdf.drop(columns=["hilbert_distance"], inplace=True)
        chunks = [
            cdf.iloc[i:i + max_cluster_size]
            for i in range(0, len(cdf), max_cluster_size)
        ]
        self.chunks = chunks
        # end
        return self
        




    
