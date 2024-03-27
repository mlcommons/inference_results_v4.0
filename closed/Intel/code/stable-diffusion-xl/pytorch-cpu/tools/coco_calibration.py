import argparse
import json
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
import tqdm
import urllib.request
import zipfile
import random
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", default="./coco-2014", help="Dataset download location"
    )
    parser.add_argument(
        "--tsv-path", default=None, help="Precomputed tsv file location"
    )
    parser.add_argument("--num-workers", default=1, type=int, help="Number of processes to download images")
    parser.add_argument(
        "--calibration-dir", default=None, help="Calibration ids location"
    )
    parser.add_argument(
        "--keep-raw", action="store_true", help="Keep the raw dataset"
    )
    parser.add_argument(
        "--download-images", action="store_true", help="Download the calibration set"
    )
    args = parser.parse_args()
    return args


def download_img(args):
    img_url, target_folder, file_name = args
    if os.path.exists(target_folder + file_name):
        log.warning(f"Image {file_name} found locally, skipping download")
    else:
        urllib.request.urlretrieve(img_url, target_folder + file_name)


if __name__ == "__main__":
    args = get_args()
    dataset_dir = args.dataset_dir
    # Convert to dataframe format and extract the relevant fields
    with open(f"{dataset_dir}/captions/captions_train2014.json") as f:
        captions = json.load(f)
        annotations = captions["annotations"]
        random_captions = random.sample(annotations, 500)
        random_captions_dict = {'caption': [data['caption'] for data in random_captions] }
        random_captions_df = pd.DataFrame(random_captions_dict)
        random_captions_df.to_csv(f"{dataset_dir}/captions/captions_source.tsv",sep="\t", index=False)

  