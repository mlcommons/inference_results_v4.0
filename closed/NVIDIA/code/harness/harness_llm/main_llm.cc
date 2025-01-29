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

#include "NvInferPlugin.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "nlohmann/json.hpp"
#include <dlfcn.h>
#include <fstream>

#include "loadgen.h"
#include "logger.h"
#include "test_settings.h"

#include "llm_server.hpp"
#include "llm_utils.hpp"
#include "qsl.hpp"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include "cuda_profiler_api.h"

using json = nlohmann::json;
using trtllmSchedulerPolicy = batch_scheduler::SchedulerPolicy;

DEFINE_string(gpu_engines, "", "Directory to the engine");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, Server, SingleStream)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "gptj", "Model name");
DEFINE_bool(use_token_latencies, false, "Report token throughput and latency information.");
DEFINE_string(trtllm_batching_mode, "IFB", "Batching mode of TRTLLM batch manager. Choices are V1 and IFB");
DEFINE_uint32(gpu_batch_size, 64, "Max Batch size to use for all devices and engines");
DEFINE_uint32(gpu_inference_streams, 1, "Number of streams for inference per device");
DEFINE_uint32(gpu_copy_streams, 1, "Number of copy streams");
DEFINE_uint32(tensor_parallelism, 1, "Tensor parallelism used in each inference group");
DEFINE_uint32(pipeline_parallelism, 1, "Pipeline parallelism used in each inference group");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(verbose_nvtx, false, "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
DEFINE_bool(enable_sort, false, "Sort the queries before sending to the batch manager");
DEFINE_bool(use_graphs, false, "TODO: enable cuda graph for the inference.");
DEFINE_string(llm_gen_config_path, "", "Path to json file storing the necessary configs for generation.");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");
DEFINE_uint64(
    server_num_issue_query_threads, 0, "Number of IssueQuery threads and RequestQueue used in Offline/Server scenario");

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
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated list if there are more than "
    "one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");

// trtllm settings
DEFINE_string(
    trtllm_batch_sched_policy, "max_util", "TRTLLM batch scheduling policy. Options are max_util and no_evict");
DEFINE_double(kvcache_free_gpu_mem_frac, 0.95, "The fraction of free gpu memory that will be used for kvcache storage");
DEFINE_bool(enable_trt_overlap, false, "Enable batches to overlap in TRT context execution");
DEFINE_string(gpu_rank_map, "",
    "GPU to rank mapping. Rank is always in increasing order from 0. Delimit GPU IDs using comma "
    "and each parallelism group with &. For example, 4 GPUs for TP2/PP1: 0,2&1,3 means GPU 0 will be mapped to rank 0, "
    "and "
    "will work with GPU 2 mapped to rank 1 for TP2/PP1. Same for GPU1 & 3 with rank 2 & 3 respectively");

// server settings
DEFINE_int32(traffic_distribution_policy, 2,
    "If not with multithreaded IssueQuery, incoming samples are distributed by this policy: "
    "0: round-robin, 1: static-chop (N samples contiguously divided by M inference groups), 3: load-balancing "
    "(inference-group"
    "that has smallest transactions on-the-fly will grab the next sample)");
DEFINE_int32(num_transactions_in_flight, kMAX_TRANSACTIONS_IN_FLIGHT, "Number of transactions in flight CommBuffer support");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {{"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream}, {"Server", mlperf::TestScenario::Server}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

std::vector<int32_t> parseDevice()
{
    std::vector<int32_t> gpus;
    int32_t numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (FLAGS_devices == "all")
    {
        LOG(INFO) << "Found " << numDevices << " GPUs";
        for (int32_t i = 0; i < numDevices; i++)
        {
            gpus.emplace_back(i);
        }
    }
    else
    {
        LOG(INFO) << "Use GPUs: " << FLAGS_devices;
        auto deviceNames = splitString(FLAGS_devices, ",");
        for (auto& n : deviceNames)
        {
            gpus.emplace_back(std::stoi(n));
        }
    }
    return gpus;
}

// 4 GPUs for TP2/PP1: 0,2&1,3 means GPU 0 will be mapped to rank 0, and will work with
// GPU 2 mapped to rank 1 for TP2/PP1. Same for GPU1 & 3 with rank 2 & 3 respectively"
std::vector<std::vector<int32_t>> parseGpuRankMapConfig(std::string const& s)
{
    std::vector<std::vector<int32_t>> map;
    if (!s.empty())
    {
        std::vector<std::string> packs = splitString(s, "&");
        for (auto& pack : packs)
        {
            std::vector<int32_t> ids;
            std::vector<std::string> nodes = splitString(pack, ",");
            for (auto& node : nodes)
            {
                ids.emplace_back(std::stoi(node));
            }
            map.emplace_back(ids);
        }
    }
    return map;
}

// Actually performs inference
int32_t doInference(int argc, char** argv)
{
    // Strip the engine directory from the engine path
    std::filesystem::path enginePath(FLAGS_gpu_engines);
    std::filesystem::path engineDir = enginePath.parent_path();
    std::filesystem::path configJsonPath = engineDir / "config.json";

    // Load the model config and generation config from the json.
    LOG(INFO) << "Loading LLM config from " << configJsonPath
              << " and geneartion config from: " << FLAGS_llm_gen_config_path;
    std::ifstream fConfig(configJsonPath.string());
    json modelConfig = json::parse(fConfig);
    std::ifstream fGenConfig(FLAGS_llm_gen_config_path);
    json genConfig = json::parse(fGenConfig);
    LLMConfig config;
    config.mMaxInputSeqlen = modelConfig["build_config"]["max_input_len"];
    config.mMaxOutputSeqlen = modelConfig["build_config"]["max_output_len"];
    config.mMaxSumSeqlen = config.mMaxInputSeqlen + config.mMaxOutputSeqlen;
    config.mEosId = genConfig["generation_config"]["eos_token_id"];
    config.mMaxGpuBatchSize = modelConfig["build_config"]["max_batch_size"];
    config.mBeamWidth = genConfig["generation_config"]["runtime_beam_width"];
    CHECK_GE(modelConfig["build_config"]["max_beam_width"], config.mBeamWidth)
        << "The engine's max_beam_width is smaller than the generation config";
    config.mTp = modelConfig["pretrained_config"]["mapping"]["tp_size"];
    config.mPp = modelConfig["pretrained_config"]["mapping"]["pp_size"];
    config.mWorldSize = config.mTp * config.mPp;

    config.mMinOutputSeqlen = genConfig["generation_config"]["min_output_len"];
    CHECK_GE(genConfig["generation_config"]["max_output_len"], config.mMaxOutputSeqlen)
        << "The engine's maxOutputSeqlen is not smaller than the generation config.";
    config.mTemperature = genConfig["generation_config"]["temperature"];
    config.mTopK = genConfig["generation_config"]["top_k"];
    config.mTopP = genConfig["generation_config"]["top_p"];
    // Streaming is enabled by the generation config flag. Let's disable it for Offline scenario
    config.mIsStreaming = genConfig["generation_config"]["streaming"] && (FLAGS_scenario != "Offline");
    config.mReportFirstToken = config.mIsStreaming;

    CHECK_EQ(config.mTp, FLAGS_tensor_parallelism)
        << "Tensor Parallelism parameter used in engine built is not matching with runtime parameter";
    CHECK_EQ(config.mPp, FLAGS_pipeline_parallelism)
        << "Pipeline Parallelism parameter used in engine built is not matching with runtime parameter";

    config.mEnableSort = FLAGS_enable_sort;
    if (FLAGS_trtllm_batching_mode == "V1")
    {
        config.mBatchMode = TrtGptModelType::V1;
    }
    else if (FLAGS_trtllm_batching_mode == "IFB")
    {
        config.mBatchMode = TrtGptModelType::InflightFusedBatching;
    }
    else
    {
        gLogError << "Unknown TRTLLM batching mode " << FLAGS_trtllm_batching_mode << ", exiting..." << std::endl;
        return EXIT_FAILURE;
    }
    if (FLAGS_trtllm_batch_sched_policy == "max_util")
    {
        config.mSchedulerPolicy = trtllmSchedulerPolicy::MAX_UTILIZATION;
    }
    else if (FLAGS_trtllm_batch_sched_policy == "no_evict")
    {
        config.mSchedulerPolicy = trtllmSchedulerPolicy::GUARANTEED_NO_EVICT;
    }
    else
    {
        gLogError << "Unknown TRTLLM batch scheduling policy " << FLAGS_trtllm_batch_sched_policy << ", exiting..."
                  << std::endl;
        return EXIT_FAILURE;
    }
    config.mKvCacheFreeGpuMemFraction = FLAGS_kvcache_free_gpu_mem_frac;
    config.mEnableTrtOverlap = FLAGS_enable_trt_overlap;

    // Configure the test settings
    mlperf::TestSettings testSettings;
    testSettings.scenario = scenarioMap[FLAGS_scenario];
    testSettings.mode = testModeMap[FLAGS_test_mode];
    //not needed from v5.0
    //testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
    testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
    testSettings.server_coalesce_queries = true;
    testSettings.server_num_issue_query_threads = FLAGS_server_num_issue_query_threads;
    testSettings.use_token_latencies = FLAGS_use_token_latencies;

    // Configure the logging settings
    mlperf::LogSettings logSettings;
    logSettings.log_output.outdir = FLAGS_logfile_outdir;
    logSettings.log_output.prefix = FLAGS_logfile_prefix;
    logSettings.log_output.suffix = FLAGS_logfile_suffix;
    logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
    logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
    logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
    logSettings.log_mode = logModeMap[FLAGS_log_mode];
    logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
    logSettings.enable_trace = FLAGS_log_enable_trace;

    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), false);
    std::vector<int32_t> gpus;

    std::string qslName(FLAGS_model + "-" + FLAGS_scenario + "-QSL");
    std::string sutName{FLAGS_model + "-" + FLAGS_scenario + "-SUT"};

    int32_t multithreadProvided;
    MPICHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &multithreadProvided));
    if (multithreadProvided < MPI_THREAD_MULTIPLE)
    {
        std::cout << "MPI multithread implementation fails to support MPI_THREAD_MULTIPLE\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int32_t numRanks = 1;
    int32_t myRank = 0;
    int32_t loadgenRank = 0;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    // Print config on one of the ranks.
    if (myRank == 0)
    {
        LOG(INFO) << config;
    }
    // Print pid for each of the rank, which makes debugging easier.
    LOG(INFO) << "Rank: " << myRank << " has pid: " << ::getpid();

    loadgenRank = numRanks - 1;
    bool const isSingleProcess = numRanks == 1;
    bool const isLoadGen = myRank == loadgenRank;
    bool const useMPI = numRanks > 1;
    int32_t const pGroupSize = config.mWorldSize;

    // NOTE: Responder can be any process of the inference group if TP
    //       Current design performs better if responder is also leader(receiver) if TP only
    //       If PP is used, it may be better to choose pGroupSize - 1 for the responder
    bool const isReceiver = (myRank % pGroupSize) == 0;
    bool const isResponder = (myRank % pGroupSize) == (config.mPp > 1 ? pGroupSize - 1 : 0);

    std::vector<std::shared_ptr<MPI_Comm>> reqComms;      // Communicator for the ranks send/recv requests
    std::vector<std::shared_ptr<MPI_Comm>> respComms;     // Communicator for the ranks send/recv responses
    std::vector<std::shared_ptr<MPI_Comm>> firstTokComms; // Communicator for the ranks send/recv first tokens
    std::vector<std::shared_ptr<MPI_Comm>>
        reclaimComms; // Communicator for the ranks dealing with reclaiming request buffers
    std::vector<std::shared_ptr<MPI_Group>> reqGrps;
    std::vector<std::shared_ptr<MPI_Group>> respGrps;
    std::vector<std::shared_ptr<MPI_Group>> firstTokGrps;
    std::vector<std::shared_ptr<MPI_Group>> reclaimGrps;

    reqComms.clear();
    respComms.clear();
    firstTokComms.clear();
    reclaimComms.clear();
    reqGrps.clear();
    respGrps.clear();
    firstTokGrps.clear();
    reclaimGrps.clear();

    auto freeCommsGroups
        = [&reqComms, &respComms, &firstTokComms, &reclaimComms, &reqGrps, &respGrps, &firstTokGrps, &reclaimGrps]() {
              // Clean up COMMs and Groups
              if (!reqComms.empty())
              {
                  for (auto& c : reqComms)
                  {
                      MPICHECK(MPI_Comm_free(c.get()));
                  }
              }
              if (!respComms.empty())
              {
                  for (auto& c : respComms)
                  {
                      MPICHECK(MPI_Comm_free(c.get()));
                  }
              }
              if (!firstTokComms.empty())
              {
                  for (auto& c : firstTokComms)
                  {
                      MPICHECK(MPI_Comm_free(c.get()));
                  }
              }
              if (!reclaimComms.empty())
              {
                  for (auto& c : reclaimComms)
                  {
                      MPICHECK(MPI_Comm_free(c.get()));
                  }
              }
              if (!reqGrps.empty())
              {
                  for (auto& g : reqGrps)
                  {
                      MPICHECK(MPI_Group_free(g.get()));
                  }
              }
              if (!respGrps.empty())
              {
                  for (auto& g : respGrps)
                  {
                      MPICHECK(MPI_Group_free(g.get()));
                  }
              }
              if (!firstTokGrps.empty())
              {
                  for (auto& g : firstTokGrps)
                  {
                      MPICHECK(MPI_Group_free(g.get()));
                  }
              }
              if (!reclaimGrps.empty())
              {
                  for (auto& g : reclaimGrps)
                  {
                      MPICHECK(MPI_Group_free(g.get()));
                  }
              }
          };

    if (FLAGS_gpu_rank_map != "" && numRanks > 1)
    {
        auto gpuRankMap = parseGpuRankMapConfig(FLAGS_gpu_rank_map);
        if (!gpuRankMap.empty())
        {
            for (auto& gpuIds : gpuRankMap)
            {
                CHECK_EQ(gpuIds.size(), pGroupSize) << "gpu_rank_map is inconsistent with TP/PP config";
                for (auto& id : gpuIds)
                {
                    gpus.emplace_back(id);
                }
            }
        }
        // It is possible gpu_rank_map does not include intentionally all the devices in the system
        // So check with the MPI world size in that case
        CHECK_EQ(gpus.size(), numRanks - 1) << "gpu_rank_map is inconsistent with runtime MPI world size";
    }
    else
    {
        gpus = parseDevice();
        CHECK_GE(gpus.size(), numRanks - 1) << "gpu_rank_map is inconsistent with runtime MPI world size";
    }
    CHECK(!gpus.empty()) << "Expecting any device here";

    // If single process or isReceiver, SUT should be LLMServerLeader: LLMServerLeader has request receiver.
    // If isResponder, SUT should be either LLMServerLeader (if isReceiver as well or isSingleProcess) or LLMServer.
    // LLMServer may not enable response sender if isResponder is set to false and handed over as an argument.
    // If multi-process and if this is LoadGen process, SUT should be LLMServerLoadGen; this doesn't have any
    // device under it, it works with LoadGen/QSL to send requests, and receive responses.
    if (isSingleProcess)
    {
        CHECK(isReceiver && isResponder) << "Receiver/Responder setup is incorrect";
        int32_t const myColor = 0;

        LOG(INFO) << "Instantiating QSL[" << qslName << "]";
        std::shared_ptr<qsl::SampleLibrary> qsl = std::make_shared<qsl::SampleLibrary>(qslName, FLAGS_map_path,
            splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, 0, true, start_from_device);
        LOG(INFO) << "Instantiated QSL[" << qslName << "]";

        LOG(INFO) << "Instantiating SUT[" << sutName << "]";
        std::shared_ptr<LLMServerLeader> sut = std::make_shared<LLMServerLeader>(sutName, qsl, engineDir, gpus, gpus,
            FLAGS_gpu_inference_streams, FLAGS_gpu_batch_size, config, testSettings.server_target_latency_percentile,
            testSettings.server_num_issue_query_threads, reqComms, respComms, firstTokComms, reclaimComms, isResponder,
            useMPI, myRank, myRank, FLAGS_num_transactions_in_flight, FLAGS_verbose_nvtx);
        LOG(INFO) << "Instantiated SUT[" << sutName << "]";

        LOG(INFO) << "Starting running actual test.";
        cudaProfilerStart();
        StartTest(sut.get(), qsl.get(), testSettings, logSettings);
        cudaProfilerStop();
        LOG(INFO) << "Finished running actual test.";

        // Need to free comms and groups within scope as TRTLLM calls MPI_Finalize()
        freeCommsGroups();
    }
    else
    {
        int32_t const numPGroups = (numRanks - 1) / pGroupSize;
        int32_t const myColor = isLoadGen ? numPGroups : myRank / pGroupSize;
        int32_t const localReceiverRank = 0;
        int32_t const localResponderRank = config.mPp > 1 ? pGroupSize - 1 : 0;

        // Make comm group for leader-loadgen/responder-loadgen
        MPI_Group worldGroup;
        MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
        for (int32_t i = 0; i < numPGroups; ++i)
        {
            // request comm group (Loadgen ranks and the leader ranks)
            if (isLoadGen || (isReceiver && (myRank / pGroupSize == i)))
            {
                int32_t ranks[2] = {pGroupSize * i + localReceiverRank, loadgenRank};
                VLOG(1) << "Rank " << myRank << ":: reqComm group ranks: (" << ranks[0] << "," << ranks[1]
                        << "); numPGroups " << numPGroups;
                reqComms.emplace_back(std::make_shared<MPI_Comm>(MPI_COMM_NULL));
                reqGrps.emplace_back(std::make_shared<MPI_Group>(MPI_GROUP_NULL));
                MPICHECK(MPI_Group_incl(worldGroup, 2, ranks, reqGrps.back().get()));
                CHECK_NE(*(reqGrps.back()), MPI_GROUP_NULL) << "Creating request Comm group failed";
                MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, *(reqGrps.back()), i, reqComms.back().get()));
                CHECK_NE(*(reqComms.back()), MPI_COMM_NULL) << "Creating request Comm group failed";
            }

            // response/first token comm group (Loadgen ranks and the last follower ranks)
            if (isLoadGen || (isResponder && (myRank / pGroupSize == i)))
            {
                int32_t ranks[2] = {pGroupSize * i + localResponderRank, loadgenRank};
                VLOG(1) << "Rank " << myRank << ":: respComm group ranks: (" << ranks[0] << "," << ranks[1]
                        << "); numPGroups " << numPGroups;
                respComms.emplace_back(std::make_shared<MPI_Comm>(MPI_COMM_NULL));
                respGrps.emplace_back(std::make_shared<MPI_Group>(MPI_GROUP_NULL));
                MPICHECK(MPI_Group_incl(worldGroup, 2, ranks, respGrps.back().get()));
                CHECK_NE(*(respGrps.back()), MPI_GROUP_NULL) << "Creating response Comm group failed";
                MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, *(respGrps.back()), i, respComms.back().get()));
                CHECK_NE(*(respComms.back()), MPI_COMM_NULL) << "Creating response Comm group failed";

                if (config.mReportFirstToken)
                {
                    VLOG(1) << "Rank " << myRank << ":: firstTokComm group ranks: (" << ranks[0] << "," << ranks[1]
                            << "); numPGroups " << numPGroups;
                    firstTokComms.emplace_back(std::make_shared<MPI_Comm>(MPI_COMM_NULL));
                    firstTokGrps.emplace_back(std::make_shared<MPI_Group>(MPI_GROUP_NULL));
                    MPICHECK(MPI_Group_incl(worldGroup, 2, ranks, firstTokGrps.back().get()));
                    CHECK_NE(*(firstTokGrps.back()), MPI_GROUP_NULL) << "Creating firstTok Comm group failed";
                    MPICHECK(
                        MPI_Comm_create_group(MPI_COMM_WORLD, *(firstTokGrps.back()), i, firstTokComms.back().get()));
                    CHECK_NE(*(firstTokComms.back()), MPI_COMM_NULL) << "Creating firstTok Comm group failed";
                }
            }

            // reclaim comm group; can be empty (when Receiver is also Responder)
            if ((!isReceiver && isResponder) || (isReceiver && !isResponder))
            {
                int32_t ranks[2] = {pGroupSize * i + localReceiverRank, pGroupSize * i + localResponderRank};
                VLOG(1) << "Rank " << myRank << ":: reclaimComm group ranks: (" << ranks[0] << "," << ranks[1]
                        << "); numPGroups " << numPGroups;
                reclaimComms.emplace_back(std::make_shared<MPI_Comm>(MPI_COMM_NULL));
                reclaimGrps.emplace_back(std::make_shared<MPI_Group>(MPI_GROUP_NULL));
                MPICHECK(MPI_Group_incl(worldGroup, 2, ranks, reclaimGrps.back().get()));
                CHECK_NE(*(reclaimGrps.back()), MPI_GROUP_NULL) << "Creating Comm group failed";
                MPICHECK(MPI_Comm_create_group(MPI_COMM_WORLD, *(reclaimGrps.back()), i, reclaimComms.back().get()));
                CHECK_NE(*(reclaimComms.back()), MPI_COMM_NULL) << "Creating Comm group failed";
            }
        }

        // Split the Comm World to support TRT-LLM's parallelism
        // Let's use TRTLLM Comm for this one, for ease of managing object lifetime used in TRTLLM
        auto& worldCommTRTLLM = tensorrt_llm::mpi::MpiComm::world();
        COMM_SESSION = worldCommTRTLLM.split(myColor, myRank);
        LOG(INFO) << "Rank " << myRank << ":: local session [rank/size | color | devId]: [" << COMM_SESSION.getRank()
                  << "/" << COMM_SESSION.getSize() << " | " << myColor << " | "
                  << (gpus.size() > myRank ? std::to_string(gpus[myRank]) : "-") << "]";

        // if this is LoadGen
        if (isLoadGen)
        {
            // This decides how many issue threads LoadGen SUT registers to LoadGen (legacy
            // server_num_issue_query_threads), and it only is valid for Server scenario. For other scenarios
            // (offline...), IssueQuery will come in from one thread (one stream), so let LoadGen SUT know about this by
            // providing useMultiIssueThreads argument
            // Let LoadGen to use multithreaded issue query if user intends to use it by setting the parameter
            bool const useMultiIssueThreads = (testSettings.scenario == mlperf::TestScenario::Server) && \
                                              (testSettings.server_num_issue_query_threads > 0);
            if (useMultiIssueThreads)
            {
                testSettings.server_num_issue_query_threads = numPGroups;
                LOG(INFO) << "Rank " << myRank << ":: Overriding server_num_issue_query_threads to " << numPGroups;
            }
            // Set up QSL/SUT for LoadGen process
            qslName += "-LoadGen";
            sutName += "-LoadGen";

            LOG(INFO) << "Rank " << myRank << ":: Started instantiating QSL[" << qslName << "]";
            std::shared_ptr<qsl::SampleLibrary> qsl = std::make_shared<qsl::SampleLibrary>(qslName, FLAGS_map_path,
                splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, 0, true, start_from_device);
            LOG(INFO) << "Rank " << myRank << ":: Finished instantiating QSL[" << qslName << "]";

            LOG(INFO) << "Rank " << myRank << ":: Started instantiating SUT[" << sutName << "]";
            std::shared_ptr<LLMServerLoadGen> sut = std::make_shared<LLMServerLoadGen>(sutName, qsl, config, numPGroups,
                useMultiIssueThreads, config.mExcludeInputInOutput, config.mEnableSort, reqComms, respComms,
                firstTokComms, myRank, FLAGS_traffic_distribution_policy, FLAGS_num_transactions_in_flight, 
                FLAGS_verbose_nvtx);
            LOG(INFO) << "Rank " << myRank << ":: Finished instantiating SUT[" << sutName << "]";

            LOG(INFO) << "Rank " << myRank << ":: Waiting for the end of init phase of other nodes...";
            // barrier waiting for initialization phase
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            LOG(INFO) << "Rank " << myRank << ":: Init phase done on all the nodes.";

            LOG(INFO) << "Rank " << myRank << ":: Starting running actual test.";
            StartTest(sut.get(), qsl.get(), testSettings, logSettings);
            LOG(INFO) << "Rank " << myRank << ":: Finished running actual test.";

            LOG(INFO) << "Rank " << myRank << ":: Initiating termination sequence.";
            sut->triggerTermination();
            LOG(INFO) << "Rank " << myRank << ":: Termination sequence initiated, waiting for others...";
            sut->waitSutCompletion();

            // barrier waiting for test finish
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            LOG(INFO) << "Rank " << myRank << ":: Sync'ed with others for shutting-off.";

            // Need to free comms and groups within scope as TRTLLM calls MPI_Finalize()
            freeCommsGroups();
            MPICHECK(MPI_Group_free(&worldGroup));
        }
        else
        {
            std::shared_ptr<qsl::SampleLibrary> qsl = nullptr;
            std::shared_ptr<LLMServer> sut = nullptr;
            sutName += " Rank" + std::to_string(myRank);

            // int32_t const myColor = myRank / pGroupSize;
            CHECK_LT(myColor, numPGroups) << "Incorrect color setting for splitting MPI Comm";

            // FIXME: Map the device based on Rank - should we use flag for this?
            std::vector<int32_t> gpuDevIds, grpDevIds;
            CHECK_LT(myRank, gpus.size());
            gpuDevIds.emplace_back(gpus[myRank]);
            for (int32_t i = 0; i < numRanks; ++i)
            {
                if (myColor == i / pGroupSize)
                {
                    grpDevIds.emplace_back(gpus[i]);
                }
            }
            CHECK_EQ(grpDevIds.size(), pGroupSize) << "Failed collecting Device IDs in the same inference group";

            LOG(INFO) << "Rank " << myRank << ":: Started instantiating SUT[" << sutName << "]";
            if (isReceiver)
            {
                sut = std::make_shared<LLMServerLeader>(sutName, qsl, engineDir, gpuDevIds, grpDevIds,
                    FLAGS_gpu_inference_streams, FLAGS_gpu_batch_size, config,
                    testSettings.server_target_latency_percentile, 1, reqComms, respComms, firstTokComms, reclaimComms,
                    isResponder, useMPI, myRank, localResponderRank, FLAGS_num_transactions_in_flight, FLAGS_verbose_nvtx);
                LOG(INFO) << "Rank " << myRank << ":: Finished instantiating SUT[" << sutName << "]";
                LOG(INFO) << "Rank " << myRank << ":: Started Leader SUT";
            }
            else
            {
                sut = std::make_shared<LLMServer>(sutName, engineDir, gpuDevIds, grpDevIds, FLAGS_gpu_inference_streams,
                    config, 1, respComms, firstTokComms, reclaimComms, isReceiver, isResponder, useMPI, myRank, FLAGS_verbose_nvtx);
                LOG(INFO) << "Rank " << myRank << ":: Finished instantiating SUT[" << sutName << "]";
                LOG(INFO) << "Rank " << myRank << ":: Started follower SUT" << (isResponder ? "; is Responder" : "");
            }

            // barrier waiting for initialization phase
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

            LOG(INFO) << "Rank " << myRank << ":: Test should start now.";
            sut->waitSutCompletion();

            // barrier waiting for test finish
            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            LOG(INFO) << "Rank " << myRank << ":: Test should finish now.";

            // Need to free comms and groups within scope as TRTLLM calls MPI_Finalize()
            freeCommsGroups();
            MPICHECK(MPI_Group_free(&worldGroup));
        }

        LOG(INFO) << "Rank " << myRank << ":: All done";
    }

    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    // initialize gLogger
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "LLM_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    gLogger.reportTestStart(sampleTest);

    // global plugin init
    gLogInfo << "Initializing TensorRT plugin library..." << std::endl;
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Call TRT-LLM function to initiate plugins
    gLogInfo << "Initializing TRT-LLM plugin libraries..." << std::endl;
    bool success = initTrtLlmPlugins(&gLogger.getTRTLogger());
    if (!success)
    {
        gLogError << "Error loading TRT-LLM plugin library, exiting..." << std::endl;
        return EXIT_FAILURE;
    }

    // Check for uintptr_t size to be sure
    CHECK_LE(sizeof(uintptr_t), sizeof(uint64_t))
        << "uintptr_t uses more than 64bit; this program will suffer undefined behavior";

    // start CUDA profiler before MPI init; need for nsys
    cudaProfilerStart();

    // perform inference
    auto retval = doInference(argc, argv);

    // NOTE: Closing MPI by calling MPI_Finalize() doesn't need as TRTLLM termination already called it
    // stop CUDA profiler after MPI_Finalize; need for nsys
    cudaProfilerStop();

    // exit
    return retval;
}
