import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        type=str,
        choices=["SingleStream", "Offline", "Server"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument("--dataset", type=str, default="coco-1024")
    parser.add_argument("--dataset-path", type=str, default="./coco2014")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--mlperf-conf", default="./configs/mlperf.conf", help="mlperf rules config")
    parser.add_argument("--user-conf", default="./configs/user.conf", help="user config for user LoadGen settings such as target QPS",)
    parser.add_argument("--audit-conf", default="audit.conf", help="audit config for LoadGen settings during compliance runs",)
    parser.add_argument("--accuracy", action="store_true", help="Enable accuracy evaluation",)
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16","bfloat16"],
        help="Data type of the model, choose from float16, and float32",
    )
    parser.add_argument("--device", type=str, choices=["cpu","xpu", "cuda"], help="xpu or cuda", default="cpu")
    parser.add_argument("--warmup", action="store_true", help="Enable warmup")
    parser.add_argument("--profile", action="store_true", help="Enable profile")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-proc", type=int, default=1, help="Number of workers")
    parser.add_argument("--cpus-per-proc", type=int, default=60, help="Number of workers")
    parser.add_argument("--workers-per-proc", type=int, default=1, help="Number of workers")
    parser.add_argument("--cores-offset", type=int, default=0, help="Cores offset")
    parser.add_argument("--verbose", action="store_true", help="Set true to dump i/o")
    parser.add_argument("--ids-path", help="Path to caption ids", default="tools/sample_ids.txt")
    parser.add_argument("--total-sample-count",type=int, default=5000)
    parser.add_argument("--model-path",type=str,help="Path of FP32 model")

    
    args = parser.parse_args()
    print(args)
    return args
