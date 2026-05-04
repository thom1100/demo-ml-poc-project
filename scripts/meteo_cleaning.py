from pathlib import Path

import pandas as pd

data = pd.read_parquet(Path.cwd().parent / "data/raw/bikes.parquet")
external_data = pd.read_csv(Path.cwd().parent / "data/raw/external_data.csv")

# Nettoyage external_data

external_data_cleaned = external_data.apply(
    lambda col: (
        col.fillna(method="ffill").fillna(method="bfill") if col.isna().any() else col
    )
)
external_data_cleaned = external_data_cleaned.dropna(axis=1, how="any")
external_data_cleaned = external_data_cleaned.drop_duplicates(subset=["date"])

full_index = pd.date_range(
    start=external_data_cleaned["date"].min(),
    end=external_data_cleaned["date"].max(),
    freq="1h",
)

full_df = pd.DataFrame(full_index, columns=["date"])

external_data_cleaned["date"] = pd.to_datetime(external_data_cleaned["date"])
full_df["date"] = pd.to_datetime(full_df["date"])

external_data_cleaned = pd.merge(full_df, external_data_cleaned, on="date", how="left")

external_data_cleaned = external_data_cleaned.interpolate("linear")

# merge des données
data["date"] = pd.to_datetime(data["date"])
external_data_cleaned["date"] = pd.to_datetime(external_data_cleaned["date"])
data_cleaned = pd.merge(data, external_data_cleaned, on="date", how="left")

# Maintenant, ne gardons que les features un minimum corrélé avec notre target
data_cleaned = data_cleaned.drop(
    columns=[
        "pres",
        "cl",
        "perssfrai",
        "rr6",
        "ssfrai",
        "pmer",
        "hnuage2",
        "rr1",
        "rr3",
        "cod_tend",
        "hnuage3",
        "w2",
        "w1",
        "nnuage4",
        "nbas",
        "hnuage4",
        "ctype2",
        "dd",
        "ctype3",
        "tend24",
        "hnuage1",
        "tend",
        "hbas",
        "cm",
        "n",
        "ht_neige",
        "ch",
        "numer_sta",
        "tminsol",
        "per",
    ]
)

data_cleaned.to_csv(("../data/processed/data_cleaned.csv"))