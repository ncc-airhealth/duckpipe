if __name__ == "__main__":

    import sys; sys.path.append("./")
    import pandas as pd
    import geopandas as gpd
    import duckpipe as dp

    DB_PATH = "example.duckdb"
    TABLE_PATH = "example.csv"
    X_COLUMN = "longitude"
    Y_COLUMN = "latitude"
    ID_COLUMN = "id"
    EPSG=4326
    MAX_CLUSTER_SIZE = 100
    MAX_CLUSTER_WIDTH = 10000
    
    df = pd.read_csv(TABLE_PATH)
    geometry = gpd.points_from_xy(df[X_COLUMN], df[Y_COLUMN])
    gdf = (
        gpd.GeoDataFrame(df, geometry=geometry, crs=EPSG)
        .loc[:, [ID_COLUMN, "geometry"]]
        .iloc[:100]
    )

    years = [2000, 2005, 2010, 2015, 2020]
    db_path = DB_PATH

    calculator = dp.Calculator(db_path=db_path, n_workers=2, memory_limit="6GB")
    geovariable = (
        calculator
        .set_dataframe(gdf)
        .chunk_by_centroid(max_cluster_size=MAX_CLUSTER_SIZE, distance_threshold=MAX_CLUSTER_WIDTH)
        .calculate_airport_distance(years=years)
        .calculate_coastline_distance(years=years)
        .calculate_landuse_area_ratio(years=years, buffer_sizes=[100, 300, 500, 1000, 5000])
        .calculate_relative_elevation(elevation_types=["dem", "dsm"], buffer_sizes=[1000, 5000])
        .get_result(pivot=True)
    )
    print(geovariable)

    geovariable.to_csv("example_result.csv", index=False)

    



