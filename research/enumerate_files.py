"""One-off: enumerate all files in the Kaggle dataset → data/file_list.txt"""
import os
import time
from pathlib import Path

os.environ["KAGGLE_CONFIG_DIR"] = str(Path(__file__).parent)

import kaggle

REF = "sadhliroomyprime/cattle-weight-detection-model-dataset-12k"
OUT = Path("data/file_list.txt")
OUT.parent.mkdir(parents=True, exist_ok=True)

api = kaggle.KaggleApi()
api.authenticate()

names = []
token = None
page = 0
while True:
    try:
        res = api.dataset_list_files(REF, page_size=200, page_token=token)
    except Exception as e:
        if "429" in str(e):
            print(f"page {page}: rate-limited, sleeping 60s …", flush=True)
            time.sleep(60)
            continue
        raise
    names.extend(f.name for f in res.files)
    token = getattr(res, "nextPageToken", None) or getattr(res, "next_page_token", None)
    page += 1
    if page % 10 == 0:
        print(f"page {page}: {len(names)} files so far", flush=True)
    if not token:
        break
    time.sleep(1.5)

OUT.write_text("\n".join(names), encoding="utf-8")
print(f"DONE: {len(names)} files written to {OUT}")
