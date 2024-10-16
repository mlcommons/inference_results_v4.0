"""
Tool to calculate accuracy for loadgen accuracy output found in mlperf_log_accuracy.json
We assume that loadgen's query index is in the same order as
the images in coco's captions/captions.tsv.
"""


import argparse
import json
import os

from PIL import Image
import numpy as np
import pandas as pd
import torch
from clip.clip_encoder import CLIPEncoder
from fid.fid_score import compute_fid



def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlperf-accuracy-file", required=True, help="path to mlperf_log_accuracy.json")
    parser.add_argument("--dataset-dir", required=True, help="coco directory")
    parser.add_argument("--statistics-path", default=None, help="path to statistics")
    parser.add_argument("--verbose", action="store_true", help="verbose messages")
    parser.add_argument("--output-file", default="coco-results.json", help="path to output file")
    parser.add_argument("--device", default="cpu", choices=["gpu", "cpu"])
    parser.add_argument("--scenario",default="offline",choices=["offline","server"])
    args = parser.parse_args()
    return args


def preprocess_image(img_dir, file_name):
    img = Image.open(img_dir + "/" + file_name)
    img = np.asarray(img)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    tensor = torch.Tensor(np.asarray(img).transpose([2,0,1])).to(torch.uint8)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3,1,1)
    return tensor.unsqueeze(0)

def main():
    args = get_args()


    annotations_file = os.path.join(args.dataset_dir, "captions", "captions.tsv")

    result_dict = {}

    # Load sample ids to save in a list
    sample_ids = [line.rstrip('\n') for line in open('./tools/sample_ids.txt', 'r')]
    image_save_path = "{}-images/".format(str(args.scenario))
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)

    # Load dataset annotations
    df_captions = pd.read_csv(annotations_file, sep="\t")

    # Load model outputs
    with open(args.mlperf_accuracy_file, "r") as f:
        results = json.load(f)

    # set device
    device = args.device if torch.cuda.is_available() else "cpu"


    # set statistics path
    statistics_path = args.statistics_path
    if args.statistics_path is None:
        statistics_path = os.path.join(os.path.dirname(__file__), "val2014.npz")

    # Load torchmetrics modules
    clip = CLIPEncoder(device=device)
    clip_scores = []
    seen = set()
    result_list = []
    with open(os.path.join(image_save_path, "captions.txt"), "w+") as caption_file:
        for j in results:
            idx = j['qsl_idx']

            if idx in seen:
                continue
            seen.add(idx)


            # Load generated image
            generated_img = np.frombuffer(bytes.fromhex(j['data']), np.uint8).reshape(1024, 1024, 3)
            result_list.append(generated_img)
            generated_img = Image.fromarray(generated_img)
            if str(idx) in sample_ids:
                image_path_tmp = image_save_path+str(idx)+".png"
                generated_img.save(image_path_tmp)


                caption_file.write(f"{idx}  {df_captions.iloc[idx]['caption']}\n")


            # Load Ground Truth
            caption = df_captions.iloc[idx]["caption"]
            clip_scores.append(
                100 * clip.get_clip_score(caption, generated_img).item()
            )
    fid_score = compute_fid(result_list, statistics_path, device)

    result_dict["FID_SCORE"] = fid_score
    result_dict["CLIP_SCORE"] = np.mean(clip_scores)
    output_file_name = "coco-results-{}.json".format(str(args.scenario))
    print(result_dict)

    with open(output_file_name, "w") as fp:
        json.dump(result_dict, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
