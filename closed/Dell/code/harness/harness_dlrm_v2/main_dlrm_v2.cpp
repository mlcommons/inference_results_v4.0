/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <dlfcn.h>
#include <thread>

#include "NvInferPlugin.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "loadgen.h"
#include "logger.h"
#include "test_settings.h"

#include "cuda_profiler_api.h"
#include "numpy.hpp"
#include "qsl.hpp"
#include "utils.hpp"

#include "dlrm_v2_qsl.h"
#include "dlrm_v2_server.h"

DEFINE_string(gpu_engines, "", "Engine");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "dlrm-v2", "Model name");
DEFINE_uint32(gpu_batch_size, 16384, "Max Batch size to use for all devices and engines");
DEFINE_bool(use_graphs, false, "Enable cudaGraphs for TensorRT engines"); // TODO: Enable support for Cuda Graphs
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(verbose_nvtx, false, "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.");

DEFINE_uint32(gpu_copy_streams, 1, "[CURRENTLY NOT USED] Number of copy streams");
DEFINE_uint32(gpu_num_bundles, 2, "Number of event-buffer bundles per GPU");
DEFINE_uint32(complete_threads, 1, "Number of threads per device for sending responses");
DEFINE_uint32(gpu_inference_streams, 1, "Number of inference streams");

DEFINE_double(warmup_duration, 1.0, "Minimum duration to run warmup for");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(server_num_issue_query_threads, 0, "Number of IssueQuery threads used in Server scenario");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// QSL arguments
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(sample_partition_path, "", "Path to sample partition file in npy format.");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated list if there are more than "
    "one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");
DEFINE_bool(start_from_device, false, "Assuming that inputs start from device memory in QSL");

// Dataset arguments
DEFINE_uint32(min_sample_size, 100, "Minimum number of pairs a sample can contain.");
DEFINE_uint32(max_sample_size, 700, "Maximum number of pairs a sample can contain.");

// BatchMaker arguments
DEFINE_uint32(num_staging_threads, 8, "Number of staging threads in DLRMv2 BatchMaker");
DEFINE_uint32(num_staging_batches, 4, "Number of staging batches in DLRMv2 BatchMaker");
DEFINE_uint32(
    max_pairs_per_staging_thread, 0, "Maximum pairs to copy in one BatchMaker staging thread (0 = use default");
DEFINE_bool(check_contiguity, false,
    "Whether to use contiguity checking in BatchMaker (default: false, recommended: true for Offline)");
DEFINE_bool(use_batcher_thread_per_device, true,
    "Instantiate BatchMaker per GPU if server_num_issue_query_threads is not used");
DEFINE_string(numa_config, "", "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices.");

// Eviction Last
DEFINE_double(eviction_last, 0.0, "Set percentage of persistance cache limit");

// DLRMv2 specific QSL limit
DEFINE_string(qsl_numa_override, "", "Designate NUMA node(s) each QSL maps to; this overrides numa config for QSL");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"Server", mlperf::TestScenario::Server},
};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {
    {"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly},
    {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {
    {"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly},
    {"Synchronous", mlperf::LoggingMode::Synchronous},
};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);

    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }

    const std::string gSampleName = "DLRMv2_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Load all the needed shared objects for plugins.
    const std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (const auto& s : plugin_files)
    {
        if (const void* dlh = dlopen(s.c_str(), RTLD_LAZY); dlh == nullptr)
        {
            gLogError << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    // Scope to force all smart objects destruction before CUDA context resets
    {
        int num_gpu = 0;
        cudaGetDeviceCount(&num_gpu);
        LOG(INFO) << "Found " << num_gpu << " GPUs";

        // Configure the test settings
        mlperf::TestSettings testSettings = {};
        testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.mode = testModeMap[FLAGS_test_mode];
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.server_coalesce_queries = true;
        testSettings.server_num_issue_query_threads = FLAGS_server_num_issue_query_threads;

        // Configure the logging settings
        mlperf::LogSettings logSettings = {};
        logSettings.enable_trace = FLAGS_log_enable_trace;
        logSettings.log_mode = logModeMap[FLAGS_log_mode];
        logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
        logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
        logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
        logSettings.log_output.outdir = FLAGS_logfile_outdir;
        logSettings.log_output.prefix = FLAGS_logfile_prefix;
        logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
        logSettings.log_output.suffix = FLAGS_logfile_suffix;

        // Gpu indices
        std::vector<int> gpus(num_gpu);
        std::iota(gpus.begin(), gpus.end(), 0);

        // Load the sample partition. We do this here to calculate the performance sample count of the underlying
        // LWIS QSL, since the super constructor must be in the constructor initialization list.
        std::vector<int> originalPartition = {};

        // Scope to automatically close the file
        {
            npy::NpyFile samplePartitionFile(FLAGS_sample_partition_path);
            CHECK_EQ(samplePartitionFile.getDims().size(), 1);

            // For now we do not allow numPartitions == FLAGS_performance_sample_count, since the TotalSampleCount
            // is determined at runtime in the underlying LWIS QSL.
            const auto numPartitions = samplePartitionFile.getDims()[0];
            CHECK_EQ(numPartitions > FLAGS_performance_sample_count, true);

            std::vector<char> samplePartitionData(samplePartitionFile.getTensorSize());
            samplePartitionFile.loadAll(samplePartitionData);

            originalPartition.resize(numPartitions);
            memcpy(originalPartition.data(), samplePartitionData.data(), samplePartitionData.size());

            LOG(INFO) << "Loaded " << originalPartition.size() - 1 << " sample partitions. ("
                      << samplePartitionData.size() << ") bytes.";
        }

        // Force underlying QSL to load all samples, since we want to be able to grab any partition given the sample
        // index.
        const size_t perfPairCount = originalPartition.back();
        const auto numaConfig = parseNumaConfig(FLAGS_numa_config);
        const auto QSLNumaConfig = parseQSLNumaConfig(FLAGS_qsl_numa_override);
        const auto numaToQslMap = getNumaToQSLMap(QSLNumaConfig);
        std::vector<DLRMv2SampleLibraryPtr_t> qsls = {};

        if (!QSLNumaConfig.empty())
        {
            CHECK(!numaConfig.empty()) << "QSL NUMA config is overriden but NUMA config is not set";

            const auto nbNumas = numaConfig.size();
            const auto numQSL = QSLNumaConfig.size();
            for (size_t qslIdx = 0; qslIdx < numQSL; ++qslIdx)
            {
                const auto constructQsl = [&]()
                {
                    bindNumaMemPolicy(QSLNumaConfig[qslIdx], nbNumas);

                    std::map<int32_t, int32_t> QSLDevMap;
                    int32_t splitIdx = 0;

                    for (const auto& numaIdx : QSLNumaConfig[qslIdx])
                    {
                        const auto myDevs = numaConfig[numaIdx].first;
                        for (auto& myDev : myDevs)
                        {
                            QSLDevMap[myDev] = splitIdx;
                            splitIdx++;
                        }
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    const auto oneQsl = std::make_shared<DLRMv2SampleLibrary>("DLRMv2 QSL", FLAGS_map_path,
                        splitString(FLAGS_tensor_path, ","), originalPartition, FLAGS_performance_sample_count,
                        perfPairCount, 0, true, FLAGS_start_from_device, QSLNumaConfig[qslIdx], QSLDevMap, nbNumas);

                    qsls.emplace_back(std::move(oneQsl));
                    resetNumaMemPolicy();
                };

                // Use a thread to construct QSL so that the allocated memory is closer to that NUMA node.
                std::thread th(constructQsl);
                bindThreadToCpus(th, numaConfig[QSLNumaConfig[qslIdx][0]].second);
                th.join();
            }
        }

        else if (!numaConfig.empty())
        {
            const auto nbNumas = numaConfig.size();
            for (size_t numaIdx = 0; numaIdx < nbNumas; numaIdx++)
            {
                const auto constructQsl = [&]()
                {
                    bindNumaMemPolicy(numaIdx, nbNumas);

                    const std::vector<int32_t> numaIndices = {static_cast<int32_t>(numaIdx)};
                    const std::map<int32_t, int32_t> QSLDevMap = {{0, 0}};

                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    const auto oneQsl = std::make_shared<DLRMv2SampleLibrary>("DLRMv2 QSL", FLAGS_map_path,
                        splitString(FLAGS_tensor_path, ","), originalPartition, FLAGS_performance_sample_count,
                        perfPairCount, 0, true, FLAGS_start_from_device, numaIndices, QSLDevMap, nbNumas);

                    qsls.emplace_back(std::move(oneQsl));
                    resetNumaMemPolicy();
                };

                // Use a thread to construct QSL so that the allocated memory is closer to that NUMA node.
                std::thread th(constructQsl);
                bindThreadToCpus(th, numaConfig[numaIdx].second);
                th.join();
            }
        }

        else
        {
            const auto oneQsl = std::make_shared<DLRMv2SampleLibrary>("DLRMv2 QSL", FLAGS_map_path,
                splitString(FLAGS_tensor_path, ","), originalPartition, FLAGS_performance_sample_count, perfPairCount,
                0, true, FLAGS_start_from_device, std::vector<int32_t>(), std::map<int32_t, int32_t>(), 0);
            qsls.emplace_back(std::move(oneQsl));
        }

        // Create server and qsl ensemble
        const auto qslEnsemble = std::make_shared<DLRMv2SampleLibraryEnsemble>(qsls);
        const auto dlrm_v2_server
            = std::make_shared<DLRMv2Server>("DLRMv2 SERVER", FLAGS_gpu_engines, qsls, gpus, FLAGS_gpu_batch_size,
                FLAGS_gpu_num_bundles, FLAGS_complete_threads, FLAGS_gpu_inference_streams, FLAGS_warmup_duration,
                FLAGS_num_staging_threads, FLAGS_num_staging_batches, FLAGS_max_pairs_per_staging_thread,
                FLAGS_check_contiguity, FLAGS_start_from_device, numaConfig, FLAGS_server_num_issue_query_threads,
                FLAGS_use_batcher_thread_per_device, FLAGS_eviction_last, numaToQslMap, FLAGS_verbose_nvtx);

        LOG(INFO) << "Starting running actual test.";
        cudaProfilerStart();
        mlperf::StartTest(dlrm_v2_server.get(), qslEnsemble.get(), testSettings, logSettings);
        cudaProfilerStop();
        LOG(INFO) << "Finished running actual test.";
    }

    return 0;
}
