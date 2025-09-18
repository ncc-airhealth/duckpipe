if __name__ == "__main__":

    import sys; sys.path.append("./")
    import pandas as pd
    from duckpipe import Calculator

    # params
    DB_PATH = "/DIRECTORY/TO/PARQUET/FILES/"
    TABLE_PATH = 'data/sample_point_korea.csv'
    X_COLUMN = "longitude"
    Y_COLUMN = "latitude"
    ID_COLUMN = "id"
    EPSG = 4326
    MAX_CLUSTER_SIZE = 100
    MAX_CLUSTER_WIDTH = 10000
    SAMPLE_SIZE = 1_000
    
    # load data, year setting
    df = pd.read_csv(TABLE_PATH).iloc[:SAMPLE_SIZE]
    years = [2000, 2005, 2010, 2015, 2020]
    
    # calculate
    calculator = Calculator(db_path=DB_PATH, n_workers=8)
    geovariable = (
        calculator
        .add_point_with_table(df, x_col=X_COLUMN, y_col=Y_COLUMN, epsg=EPSG)
        .chunk_by_centroid(max_cluster_size=MAX_CLUSTER_SIZE, distance_threshold=MAX_CLUSTER_WIDTH)
        .calculate_coordinate()
        .calculate_airport_distance(years=years)
        .calculate_coastline_distance(years=years)
        .calculate_landuse_area_ratio(years=years, buffer_sizes=[100, 300, 500, 1000, 5000])
        .calculate_relative_elevation(elevation_types=["dem", "dsm"], buffer_sizes=[1000, 5000])
        .calculate_main_road_distance(mr_types=['mr1', 'mr2'], years=[2005, 2010, 2015, 2020])
        .calculate_road_distance(years=[2005, 2010, 2015, 2020])
        .calculate_road_llw(buffer_sizes=[25, 50, 100, 300, 500, 1000, 5000], years=[2020])
        .get_result(pivot=True)
    )
    print(geovariable)