import pandas as pd
import geopandas as gpd

df0 = pd.read_csv("data/sample_point_korea.csv")
df1 = (
    gpd.GeoSeries.from_xy(df0['x'], df0['y'], crs='EPSG:5179')
    .reset_index(name='geometry')
    .drop('index', axis=1)
    .assign(hilbert=lambda df: df.geometry.hilbert_distance())
    .to_crs('EPSG:4326')
    .assign(geometry=lambda df: df.geometry.set_precision(1e-6))
    .sort_values(by='hilbert', ascending=False)
    .drop('hilbert', axis=1)
    .reset_index(drop=True)
    .reset_index(names='id')
    .assign(longitude=lambda df: df.geometry.x)
    .assign(latitude=lambda df: df.geometry.y)
    .drop('geometry', axis=1)
)
df1.to_csv("data/sample_point_korea2.csv", index=False)
print(df1)