//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#ifndef LOADGEN_WRAPPER_INTF_H_
#define LOADGEN_WRAPPER_INTF_H_

#include <future>
#include "loadgen_wrapper.h"

using namespace simaai::mlperf_wrapper;

/**
 * @brief LoadgenWrapper Singleton class called by gstreamer src/sink plugins
 */
class LoadgenSingleton {
 public:
 private:
  /**
   * Singleton instance of Loadgen singleton wrapper
   */
  static LoadgenSingleton * loadgen;
  /**
   * Mutex to protect access to singleton class
   */
  static std::mutex m;

  LoadgenSingleton(){};
  virtual ~LoadgenSingleton() = default;
  LoadgenSingleton(const LoadgenSingleton&) = delete;

  std::once_flag loadgen_flag;

 public:
  LoadgenWrapper<int8_t> loader;
  LoadgenConfig cfg;

  /**
   * @brief Initialize loadenwrapper from the singleton
   * This api is thread safe to be called from multiple threads
   * @param[in] LoadgenConfiguration passed to loadgenwrapper
   */
  void loadgen_init_impl (LoadgenConfig & cfg) {
    std::call_once(loadgen_flag, [&cfg, this]() {
      loader = LoadgenWrapper<int8_t>(cfg);
      if (!loadgen_start_impl())
        std::cout << "--- MLPerf:Loadgen Error\n";
      std::cout << "Done with start\n";
    });
  }

  /**
   * @brief Start the loadgen, loads the image data into dram
   */
  bool loadgen_start_impl () {
    loader.loadgen_start();
    std::cout << "Returning from loadgen start\n";
    return true;
  }

  /**
   * @brief Helper API to get the current batchsize
   */
  size_t loadgen_get_cur_bs () {
    return loader.get_current_batch_size();
  }

  /**
   * @brief Helper API wrapper to get the current sample based on loaded samples
   * @param[out] out param of writing backt he number of bytes loaded
   */
  void loadgen_get_sample(size_t * bytes, int8_t * data) {
    uint64_t addr[8];
    loader.loadgen_get_sample(bytes, addr);
    if (data == NULL)
      std::cout << "--- MLPerf::ERROR Data is null please check loader\n";

    return;
  }

  /**
   * @brief Helpder API to get the current sample idx
   * @return Returns the id for bs=1
   */
  int loadgen_get_current_sample_index(int idx) {
    return loader.get_current_sample_index(idx);
  }

  /**
   * @brief Wrapper to loadgen_complete sample, which completes the current sample so that we get the next one
   * @param arg_max the arg_max calculted from the sink plugin
   */
  void loadgen_complete_sample(const std::vector<int32_t> & arg_max) {
    loader.loadgen_complete_sample(arg_max);
  }

  /**
   * Default delete copy and get instance operator
   */
  void operator=(const LoadgenSingleton &) = delete;

  static LoadgenSingleton *get_instance();
};

#endif // LOADGEN_WRAPPER_INTF_H_
