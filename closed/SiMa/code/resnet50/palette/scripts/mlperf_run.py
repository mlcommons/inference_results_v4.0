#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
# Code owner: Vimal Nakum
#########################################################
"""
Given a network which produces a combination of .lm and .so
files when compiled, run these files on the DevKit and emit
FPS KPIs to the console.
"""
import os
import argparse
from os import path
from typing import List, Dict
import time
import json
import warnings
import numpy as np
import sys
import os
import logging
import getpass
import json
import subprocess
import glob
import shutil

# Power server definitions
NTP_SERVER="time.google.com"
POWER_SERVER="192.168.89.82"

# Batch size for various MLPerf runs
SINGLE_STREAM = 1
MULTI_STREAM = 8
OFFLINE = 14

# Run types
PERFORMANCE_RUN = 0
ACCURACY_RUN = 1

class MLPerfCompliance:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_path = args.model_path
        self.resnet_path = args.resnet_path
        self.libs_path = args.libs_path
        self.inf_comp_path = args.inf_comp_path
        self.output_path = args.output_path
        # create directory structure for tests and logs
        self.outputdir_name = "mlperf_run_" + time.strftime("%Y%m%d-%H%M%S")
        return

    def copy_mlperf_files(self, dest_dir) -> None:
        # copy the logs to log directory path
        for filename in glob.glob("mlperf_log*"):
            source_file = os.path.join(os.getcwd(), filename)  # Construct full source path
            dest_file =  os.path.join(dest_dir, filename)  # Construct full destination path
            shutil.move(source_file, dest_file)
            #shutil.copy2(source_file, dest_file)
            print(f"Moved {filename} to {dest_dir}")
        if os.path.exists("accuracy.txt"):
            source_file = os.path.join(os.getcwd(), "accuracy.txt")  # Construct full source path
            dest_file =  os.path.join(dest_dir, "accuracy.txt")  # Construct full destination path
            shutil.move(source_file, dest_file)
            print(f"Moved {source_file} to {dest_dir}")
            
        return

    # returns mlperf scenario for the batch size
    def get_scenario_from_batch_size(self, batchsize) -> int:
        if batchsize == SINGLE_STREAM:
            scenario = 0
        elif batchsize == MULTI_STREAM:
            scenario = 1
        else:
            scenario = 2
        return scenario

    # Executes run2 script that would run the mlperf workload and save the results in mlperf_log* files
    def execute_run2_script(self, batchsize, run_type) -> None:
        scenario = self.get_scenario_from_batch_size(batchsize)
            
        # prepare run2 command string with provided parameters
        # ./run2 -t 0 -i "1:3:224:224" -o "1:64" -r 0 -s 0 -b 1 -d /mnt/vimal.nakum/mlperf/resnet_dataset/mlperf_resnet50_dataset.dat -m "false" -l /mnt/vimal.nakum/mlperf/libs_0126/ -a /mnt/vimal.nakum/mlperf/models_2024_01_11
        run2_cmd_str = "./run2 -t 0 -i " + str(batchsize) + ":3:224:224 -o " + str(batchsize) + ":64 -r " + str(run_type) + " -s " + str(scenario) + " -b " + str(batchsize) + " -d " + self.resnet_path + " -m false" + " -l " + self.libs_path + " -a " + self.model_path
        print(f"run2 string {run2_cmd_str}")
        result = subprocess.run(run2_cmd_str, shell=True)
        if result.returncode != 0:
            print(f"run2_cmd {run2_cmd_str} failed")
 
        return result.returncode

    # Runs a single test for a given test, using its audit.config file for a given batch size
    # It returns directory names for normal and audit runs that can be used to run
    # test specific verification scripts
    def run_single_test(self, test_name:str, batchsize:int, audit_config_path:str) -> None:

        # Submission directories - All these directories will have TEST01/TEST04 and TEST05 directories
        # Log directories for reference results and logs
        if batchsize == SINGLE_STREAM:
            sub_dir_path = os.getcwd() + "/" + self.outputdir_name + "/closed/SiMa/compliance/davinci_hhhl/resnet50/"+ "SingleStream"
            log_normal_perf_dir_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/SingleStream/normal/" + test_name + "/performance/run_1"
            log_normal_acc_dir_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/SingleStream/normal/" + test_name + "/accuracy"
            log_audit_perf_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/SingleStream/audit/" + test_name + "/performance/run_1"
            log_normal_test_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/SingleStream/normal/" + test_name
            log_audit_test_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/SingleStream/audit/" + test_name
        elif batchsize == MULTI_STREAM:
            sub_dir_path = os.getcwd() + "/" + self.outputdir_name + "/closed/SiMa/compliance/davinci_hhhl/resnet50/"+ "MultiStream"
            log_normal_perf_dir_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/MultiStream/normal/" + test_name + "/performance/run_1"
            log_normal_acc_dir_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/MultiStream/normal/" + test_name + "/accuracy"
            log_audit_perf_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/MultiStream/audit/" + test_name + "/performance/run_1"
            log_normal_test_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/MultiStream/normal/" + test_name
            log_audit_test_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/MultiStream/audit/" + test_name
        else:
            sub_dir_path = os.getcwd() + "/" + self.outputdir_name + "/closed/SiMa/compliance/davinci_hhhl/resnet50/"+ "Offline"
            log_normal_perf_dir_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/Offline/normal/" + test_name + "/performance/run_1"
            log_normal_acc_dir_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/Offline/normal/" + test_name + "/accuracy"
            log_audit_perf_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/Offline/audit/" + test_name + "/performance/run_1"
            log_normal_test_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/Offline/normal/" + test_name
            log_audit_test_path = os.getcwd() + "/" + self.outputdir_name + "/run_logs/Offline/audit/" + test_name

        os.makedirs(sub_dir_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists
        os.makedirs(log_normal_perf_dir_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists
        os.makedirs(log_normal_acc_dir_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists
        os.makedirs(log_audit_perf_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists
        os.makedirs(log_audit_test_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists

        # Run normal test
        self.execute_run2_script(batchsize, PERFORMANCE_RUN)
        if not os.path.exists("mlperf_log_summary.txt"):
            print(f"mlperf run failed - please check the log files for details")
            return None
        self.copy_mlperf_files(log_normal_perf_dir_path)

        self.execute_run2_script(batchsize, ACCURACY_RUN)
        if not os.path.exists("mlperf_log_summary.txt"):
            print(f"mlperf run failed - please check the log files for details")
            return None
        self.copy_mlperf_files(log_normal_acc_dir_path)

        # Run audit test
        # copy audit config file from mlperf mlcommons for test01
        if not os.path.exists(audit_config_path):
            print(f"{test_name} has no audit config file at {audit_config_path}")
            return None
        
        shutil.copy(audit_config_path, ".")

        self.execute_run2_script(batchsize, PERFORMANCE_RUN)
        if not os.path.exists("mlperf_log_summary.txt"):
            print(f"mlperf run failed - please check the log files for details")
            return None
        self.copy_mlperf_files(log_audit_perf_path)

        self.execute_run2_script(batchsize, ACCURACY_RUN)
        if not os.path.exists("mlperf_log_summary.txt"):
            print(f"mlperf run failed - please check the log files for details")
            return None
        self.copy_mlperf_files(log_audit_test_path)

        # delete the run.log files as they are not needed now
        file_list = glob.glob(os.path.join(os.getcwd(), "run.log*"))
        for filename in file_list:
            try:
                os.remove(filename)
                print(f"Deleted file: {filename}")
            except OSError as e:
                if e.errno != 2:
                    print(f"Error deleting {filename}:", e)

        return log_normal_test_path, log_audit_test_path, sub_dir_path

    # Runs TEST01 for SingleStream, MultiStream and Offline
    # Run a normal test and save the results for accuracy and performance
    # Run same tests with audit.config file from mlcommons
    # Run mlcommons verification script with the above results
    def run_test01(self, batchsizes:List) -> bool:

        # Run the TEST01 in both normal and audit mode
        file_path = "compliance/nvidia/" + "TEST01" + "/resnet50/audit.config"
        audit_config_path = os.path.join(self.inf_comp_path, file_path)
        for batchsize in batchsizes:
            print(f"Running TEST01 batchsize {batchsize}")
            result = self.run_single_test("TEST01", batchsize, audit_config_path)
            if result is None:
                print(f"Running TEST01 failed")
                return None
            
            log_normal_test_path, log_audit_test_path, sub_dir_path = result

            # Run verification scripts
            cmd_exec_path = os.path.join(self.inf_comp_path, "compliance/nvidia/TEST01/run_verification.py")
            cmd_exec_str = "python3 " + cmd_exec_path + " --results_dir " + log_normal_test_path + " --compliance_dir " + log_audit_test_path + " --output_dir " + sub_dir_path
            print(f"test01 run string {cmd_exec_str}")
            result = subprocess.run(cmd_exec_str, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"test01 mlperf verification {cmd_exec_str} failed")
                return False
 
            outs = result.stdout
            errs = result.stderr
            if not ("Accuracy check pass: True" in outs and "Performance check pass: True" in outs):
                print(f"test01 mlperf verification {cmd_exec_str} did not pass")
                return False

            # python3 /project/davinci_users/software/vimal.nakum/src/mlperf/inference/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --imagenet-val-file /project/davinci_users/software/vimal.nakum/src/mlperf/test_submissions/SiMa_test_0201/closed/SiMa/code/resnet50/palette/data/val_map.txt > accuracy.txt
            # TODO - generate accuracy measurement file - accuracy.txt
            # Run accuracy checker from mlperf_log_accuracy.json and output accuracy.txt
            accuracy_dir_path = sub_dir_path + "/TEST01/accuracy/"
            if not self.run_inference_accuracy_checker(accuracy_dir_path, accuracy_dir_path):
                print(f"run_inference_accuracy_checker failed")
                return False

            # delete normal and audit logs, as they take too much space
            shutil.rmtree(log_normal_test_path)
            shutil.rmtree(log_audit_test_path)

        return True

    # Runs TEST04 for SingleStream, MultiStream and Offline
    # Run a normal test and save the results for accuracy and performance
    # Run same tests with audit.config file from mlcommons
    # Run mlcommons verification script with the above results
    def run_test04(self, batchsizes:List) -> bool:

        # Run the TEST01 in both normal and audit mode
        file_path = "compliance/nvidia/" + "TEST04" + "/audit.config"
        audit_config_path = os.path.join(self.inf_comp_path, file_path)
        for batchsize in batchsizes:
            print(f"Running TEST04 batchsize {batchsize}")
            log_normal_test_path, log_audit_test_path, sub_dir_path = self.run_single_test("TEST04", batchsize, audit_config_path)

            # Run verification scripts
            cmd_exec_path = os.path.join(self.inf_comp_path, "compliance/nvidia/TEST04/run_verification.py")
            cmd_exec_str = "python3 " + cmd_exec_path + " --results_dir " + log_normal_test_path + " --compliance_dir " + log_audit_test_path + " --output_dir " + sub_dir_path
            print(f"test04 run string {cmd_exec_str}")
            result = subprocess.run(cmd_exec_str, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"test04 mlperf verification {cmd_exec_str} failed")
                return False
            outs = result.stdout
            errs = result.stderr

            if not "Performance check pass: True" in outs:
                print(f"test04 mlperf verification {cmd_exec_str} did not pass")
                return False
            
            # delete normal and audit logs, as they take too much space
            shutil.rmtree(log_normal_test_path)
            shutil.rmtree(log_audit_test_path)

        return True

    # Runs TEST05 for SingleStream, MultiStream and Offline
    # Run a normal test and save the results for accuracy and performance
    # Run same tests with audit.config file from mlcommons
    # Run mlcommons verification script with the above results
    def run_test05(self, batchsizes:List) -> None:

        # Run the TEST01 in both normal and audit mode
        file_path = "compliance/nvidia/" + "TEST05" + "/audit.config"
        audit_config_path = os.path.join(self.inf_comp_path, file_path)
        for batchsize in batchsizes:
            print(f"Running TEST05 batchsize {batchsize}")
            log_normal_test_path, log_audit_test_path, sub_dir_path = self.run_single_test("TEST05", batchsize, audit_config_path)

            # Run verification scripts
            cmd_exec_path = os.path.join(self.inf_comp_path, "compliance/nvidia/TEST05/run_verification.py")
            cmd_exec_str = "python3 " + cmd_exec_path + " --results_dir " + log_normal_test_path + " --compliance_dir " + log_audit_test_path + " --output_dir " + sub_dir_path
            print(f"test05 run string {cmd_exec_str}")
            result = subprocess.run(cmd_exec_str, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"test05 mlperf verification {cmd_exec_str} failed")
                return False

            outs = result.stdout
            errs = result.stderr

            if not "Performance check pass: True" in outs:
                print(f"test05 mlperf verification {cmd_exec_str} did not pass")
                return False
            
            # delete normal and audit logs, as they take too much space
            shutil.rmtree(log_normal_test_path)
            shutil.rmtree(log_audit_test_path)
            
        return True

    # Delete previous runs files - mlperf_log*, audit.config, run.log*, run_mlart.log
    def delete_old_run_files(self) -> None:
        files_to_del_list = []
        file_list = glob.glob(os.path.join(os.getcwd(), "mlperf_log*"))
        files_to_del_list = files_to_del_list + file_list
        if os.path.exists("./audit.config"):
            files_to_del_list.append("./audit.config")
        file_list = glob.glob(os.path.join(os.getcwd(), "run.log*"))
        files_to_del_list = files_to_del_list + file_list
        if os.path.exists("./run_mlart.log"):        
            files_to_del_list.append("./run_mlart.log")
        
        for filename in files_to_del_list:
            try:
                os.remove(filename)
                print(f"Deleted file: {filename}")
            except OSError as e:
                if e.errno != 2:
                    print(f"Error deleting {filename}:", e)
        return
            
    # Runs all MLPerf required tests
    def run_mlperf_compliance_tests(self, batchsizes:List, comp_test:str) -> bool:
        
        # delete old files    
        self.delete_old_run_files()
        
        print("Running MLPerf Compliance tests")
        # Run all the tests
        if comp_test == "all":
            print(f"Running All tests")
            if not self.run_test01(batchsizes):
                return False
            if not self.run_test04(batchsizes):
                return False
            if not self.run_test05(batchsizes):
                return False
        elif comp_test == "TEST01":
            if not self.run_test01(batchsizes):
                return False
        elif comp_test == "TEST04":
            if not self.run_test04(batchsizes):
                return False
        elif comp_test == "TEST05":
            if not self.run_test05(batchsizes):
                return False
            
        return True

    # Runs inference accuracy checker and outputs accuracy.txt file
    def run_inference_accuracy_checker(self, acc_json_dir_path:str, output_acc_dir_path:str) -> bool:

        # python3 /project/davinci_users/software/vimal.nakum/src/mlperf/inference/vision/classification_and_detection/tools/accuracy-imagenet.py --mlperf-accuracy-file ./mlperf_log_accuracy.json --imagenet-val-file /project/davinci_users/software/vimal.nakum/src/mlperf/test_submissions/SiMa_test_0201/closed/SiMa/code/resnet50/palette/data/val_map.txt > accuracy.txt
        # generate accuracy measurement file - accuracy.txt
        accuracy_script_path = self.inf_comp_path + "/vision/classification_and_detection/tools/accuracy-imagenet.py"
        accuracy_cmd_str = "python3 " + accuracy_script_path + " --mlperf-accuracy-file " + acc_json_dir_path + "mlperf_log_accuracy.json " + " --dtype int32 " + "--imagenet-val-file ./val_map.txt > " + output_acc_dir_path + "accuracy.txt"
        result = subprocess.run(accuracy_cmd_str, shell=True)
        if result.returncode != 0:
            print(f"check accuracy command {accuracy_cmd_str} failed")
            return False
        return True

    # Runs power test for Performance results and Accuracy tests
    def run_power_tests(self, batchsizes:List) -> bool:

        print(f"Running Power tests")
        output_dir = os.getcwd() + "/" + self.outputdir_name + "/closed/SiMa/results/davinci_hhhl/resnet50/"
        
        for batchsize in batchsizes:
            scenario = self.get_scenario_from_batch_size(batchsize)

            print(f"Running Power Test batchsize {batchsize}")

            if batchsize == SINGLE_STREAM:
                sub_dir_path = output_dir + "SingleStream/performance"
                acc_dir_path = output_dir + "SingleStream/accuracy"
            elif batchsize == MULTI_STREAM:
                sub_dir_path = output_dir + "MultiStream/performance"
                acc_dir_path = output_dir + "MultiStream/accuracy"
            else:
                sub_dir_path = output_dir + "Offline/performance"
                acc_dir_path = output_dir + "Offline/accuracy"

            os.makedirs(sub_dir_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists
            os.makedirs(acc_dir_path, exist_ok=True)  # `exist_ok` prevents errors if a directory already exists
            
            # Store power logs temporarily in temp directory, as it can not be any subdirectory of the
            # current directory
            os.makedirs("../temp", exist_ok=True)

            # Run in performance mode
            run2_cmd_str = "./run2 -t 0 -i " + str(batchsize) + ":3:224:224 -o " + str(batchsize) + ":64 -r 0 " + " -s " + str(scenario) + " -b " + str(batchsize) + " -d " + \
                    self.resnet_path + " -m false" + " -l " + self.libs_path + " -a " + self.model_path

            # run power command - it would copy the results with power and performance logs in ../temp
            ptd_cmd_str = f'/mnt/power-dev/ptd_client_server/client.py --addr "{POWER_SERVER}" --output "../temp" \
    --run-workload "{run2_cmd_str}" --loadgen-logs "." --label "singlestream" --ntp "{NTP_SERVER}"'
            print(f"ptd cmd str {ptd_cmd_str}")
            
            result = subprocess.run(ptd_cmd_str, shell=True)
            if result.returncode != 0:
                print(f"ptd command {ptd_cmd_str} failed")
                return False

            # copy the log files from temp directory
            source_dir = "temp/"+ next(os.walk("../temp"))[1][0]
            
            power_log_dir = os.path.join(os.path.dirname(os.getcwd()), source_dir)

            # remove ptd_out.txt
            ptd_out_file_path = os.path.join(power_log_dir, "run_1/ptd_out.txt")
            os.remove(ptd_out_file_path)

            print(f"copy power dir {power_log_dir} to {sub_dir_path}")
            shutil.copytree(power_log_dir, sub_dir_path, dirs_exist_ok=True)
            shutil.rmtree("../temp")
            
            # Run the accuracy test
            self.execute_run2_script(batchsize, ACCURACY_RUN)
            if not os.path.exists("mlperf_log_summary.txt"):
                print(f"mlperf run failed - please check the log files for details")
                return False
            
            # Run accuracy checker from local mlperf_log_accuracy.json and output accuracy.txt
            # in the same dir
            if not self.run_inference_accuracy_checker("./", "./"):
                print(f"run_inference_accuracy_checker failed")
                return False

            self.copy_mlperf_files(acc_dir_path)

        return True
        
    def run_mlperf_result_tests(self, batchsizes:List) -> bool:

        # delete old files    
        self.delete_old_run_files()

        # Run the performance tests with power and accuracy test
        if not self.run_power_tests(batchsizes):
            print(f"running power tests failed")
            return False

        return True
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute MLPerf tests with required output directory structure')
    parser.add_argument('--model_path', required=True, type=str, help='Path to mlperf models directory')
    parser.add_argument('--resnet_path', required=True, type=str, help='Path to resnet dataset')
    parser.add_argument('--libs_path', required=True, type=str, help='Path to mlperf libs')
    parser.add_argument('--inf_comp_path', required=True, type=str, help='Path to mlcommons compliance directory')
    parser.add_argument('--output_path', required=True, type=str, help='Path to store output directory')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument('--batchsize', required=False, type=int, help='Batch size to run, default all')
    parser.add_argument('--comp_test', required=False, type=str, default="all", help='Compliance test - TEST01, TEST04, TEST05')
    parser.add_argument('--run_power', required=False, type=bool, default=False, help='Compliance test - TEST01, TEST04, TEST05')
    
    args = parser.parse_args()

    # check that all paths are valid
    if not os.path.exists(args.model_path):
        print(f"Invalid model path {args.model_path}")
        sys.exit(-1)
        
    if not os.path.exists(args.resnet_path):
        print(f"Invalid resnet dataset path {args.resnet_path}")
        sys.exit(-1)

    if not os.path.exists(args.libs_path):
        print(f"Invalid mlperf lib path {args.lib_path}")
        sys.exit(-1)

    if not os.path.exists(args.inf_comp_path):
        print(f"Invalid MLPerf inference compliance path {args.inf_comp_path}")
        sys.exit(-1)

    if not os.path.exists(args.output_path):
        print(f"Invalid MLPerf output path {args.output_path}")
        sys.exit(-1)
        
    # check if val_map.txt is present in current directory
    if not os.path.exists("./val_map.txt"):
        print(f"missing val_map.txt in {os.getcwd()} - please copy from arch_mlperf/data/val_map.txt")
        sys.exit(-1)

    if args.verbose:
        # Create the logfile with BasicConfig
        logging.basicConfig(filename='mlperf_run.log', level=logging.DEBUG)
        logging.debug('This message should go to the log file')
        
    # create mlperf test object
    mlperf_comp_test = MLPerfCompliance(args)
    
    if args.comp_test not in ["TEST01", "TEST04", "TEST05", "all"]:
        print(f"Wrong test {args.comp_test} - valid tests are TEST01, TEST04, TEST05 or all")
        sys.exit(-1)

    if args.run_power:
        print(f"Going to run power tests")

    # run the tests
    if not args.batchsize:
        batchsizes = [SINGLE_STREAM, MULTI_STREAM, OFFLINE]
    elif args.batchsize == 1:
        batchsizes = [SINGLE_STREAM]
    elif args.batchsize == 8:
        batchsizes = [MULTI_STREAM]
    elif args.batchsize == 14:
        batchsizes = [OFFLINE]
    else:
        print(f"Wrong batch size {args.batchsize} only 1/8/14 are supported")
        sys.exit(-1)
        
    # run the power tests
    if args.run_power:
        if not mlperf_comp_test.run_mlperf_result_tests(batchsizes):
            print(f"run_mlperf_result_tests failed")
            sys.exit(-1)

    if not mlperf_comp_test.run_mlperf_compliance_tests(batchsizes, args.comp_test):
        print(f"run_mlperf_compliance_tests failed")
        sys.exit(-1)

    # Truncate accuracy.json files and remove backup dir afterwards
    print(f"Truncating accuracy.json files")
    # truncate the accuracy.json by - python3 /project/davinci_users/software/vimal.nakum/src/mlperf/inference/tools/submission/truncate_accuracy_log.py --input SiMa_test_0201/ --submitter SiMa --backup ../
    if os.path.exists("temp"):
        print(f"remove temp dir as it would be created during truncation")
        shutil.rmtree("temp")

    truncate_file_path = os.path.join(args.inf_comp_path, "tools/submission/truncate_accuracy_log.py")
    truncate_cmd_str = "python3 " + truncate_file_path + " --input " + mlperf_comp_test.outputdir_name + " --submitter SiMa --backup ../temp"
    print(f"truncate cmd str - {truncate_cmd_str}")
    result = subprocess.run(truncate_cmd_str, shell=True)
    if result.returncode != 0:
        print(f"ptd command {truncate_cmd_str} failed")
        sys.exit(-1)

    print(f"Test successful - logs are stored in {mlperf_comp_test.outputdir_name}")
            
