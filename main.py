from data.polygon_agg_download import download_aggregates
from data.storage import save_parquet
from data.storage import save_csv
from features.build_features import build_features

API_KEY = "mNR4GsYtuDWZWajAM24pyQHpOU3f56EI"  # rotate the old one

def main():
    df = download_aggregates(
        symbol="AAPL",
        multiplier=1,
        timespan="minute",
        start="2024-01-01",
        end="2024-02-01",
        api_key=API_KEY,
    )

    path = save_csv(
        df,
        "raw/stocks/AAPL_1m_2024-01.csv"
    )

    feats = build_features(df)

    out_path = save_csv(feats.reset_index(), "processed/stocks/AAPL_1m_features_2024-01.csv")
    print("Saved features to:", out_path)

    print("Saved to:", path)


if __name__ == "__main__":
    main()