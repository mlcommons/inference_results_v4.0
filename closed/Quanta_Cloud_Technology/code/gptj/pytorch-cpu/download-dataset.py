import os
import sys
import json
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import load_dataset

set_id='cnn_dailymail'
version='3.0.0'
instruction_template="Summarize the following news article:"

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="validation", help="Split to use")
    parser.add_argument("--output-dir", help="Output directory")

    return parser.parse_args()

def check_path(path):
    return os.path.exists(path)

def prepare_calibration_data(split, output_dir):

    dataset = load_dataset("cnn_dailymail", name="3.0.0", split=split)
    train = dict((x['id'], x) for x in dataset)
    
    inputs = []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        x = dict()
        x["instruction"] = instruction_template
        x["input"] = sample["article"]
        x["output"] = sample["highlights"]
        inputs.append(x)
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir,"cnn_dailymail_{}.json".format(split))
    with open(output_path, 'w') as write_f:
        json.dump(inputs, write_f, indent=4, ensure_ascii=False)

    print("{} data saved at {}".format(split, output_path))

def main():

    args = get_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "cnn-dailymail-{}".format(args.split)
    prepare_calibration_data(args.split, output_dir)

if __name__=="__main__":
    main()

