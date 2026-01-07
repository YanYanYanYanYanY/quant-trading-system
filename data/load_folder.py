import pandas as pd
from pathlib import Path

def load_file(path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a single file into a DataFrame based on file extension.
    Extra kwargs are passed to pandas readers.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path, **kwargs)

    elif ext == ".parquet":
        return pd.read_parquet(path, **kwargs)

    elif ext == ".feather":
        return pd.read_feather(path, **kwargs)

    elif ext == ".json":
        return pd.read_json(path, **kwargs)

    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, **kwargs)

    elif ext == ".pkl":
        return pd.read_pickle(path)

    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    
    
def load_folder(
    folder_path: str,
    add_symbol: bool = True,
    date_col: str | None = None,
) -> pd.DataFrame:
    """
    Load all supported files in a folder and combine into one DataFrame.
    """
    folder = Path(folder_path)
    files = sorted(folder.iterdir())

    dfs = []

    for f in files:
        if not f.is_file():
            continue

        try:
            df = load_file(f)

            if add_symbol:
                df["symbol"] = f.stem

            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])

            dfs.append(df)

        except ValueError:
            # unsupported format
            continue
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not dfs:
        raise RuntimeError("No valid data files loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined