import pandas as pd

bike_df = pd.read_parquet("../data/raw/bikes.parquet")
weather_df = pd.read_csv("../data/raw/weather_data.csv")

bike_df["date"] = pd.to_datetime(bike_df["date"]).dt.date
weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

final_df = bike_df.merge(weather_df, on="date", how="left")
print(final_df.head())

final_df.to_csv("../data/processed/df_cleaned.csv")