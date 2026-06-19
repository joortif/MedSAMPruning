import argparse
from pathlib import Path
import requests

DATASETS = {
    "BUSI": 20538442,
    "MMOTU": 20541610,
}

def download_dataset(dataset_name: str, outdir: str):
    dataset_name = dataset_name.upper()

    if dataset_name not in DATASETS:
        raise ValueError(f"Invalid dataset: {dataset_name}. Options are BUSI or MMOTU.")

    record_id = DATASETS[dataset_name]

    meta_resp = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    root = Path(outdir).expanduser() if outdir is not None else Path(__file__).resolve().parents[2] / "data"
    out_path = root / dataset_name
    out_path.mkdir(parents=True, exist_ok=True)

    for file_info in meta.get("files", []):
        file_name = file_info["key"]
        download_url = file_info["links"]["self"]

        dest = out_path / file_name

        print(f"Downloading {file_name}...")
        with requests.get(download_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"Saved in: {dest.resolve()}")