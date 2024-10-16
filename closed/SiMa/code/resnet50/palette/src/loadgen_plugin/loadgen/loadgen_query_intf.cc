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

#include <iostream>

#include "loadgen_query_intf.h"
#include "loadgen_wrapper_ctrl.h"

std::condition_variable LoadgenWrapperCtrl::query_cv;
std::mutex LoadgenWrapperCtrl::query_mutex;
size_t LoadgenWrapperCtrl::last_seen;
size_t LoadgenWrapperCtrl::next_example_ready;
size_t LoadgenWrapperCtrl::next_example_size;
std::vector<mlperf::ResponseId> LoadgenWrapperCtrl::next_example_ids;
std::vector<int64_t> LoadgenWrapperCtrl::cur_ids;
std::atomic<int> LoadgenWrapperCtrl::is_mlperf_done;
std::future<int> LoadgenWrapperCtrl::future;

// Called when loadgen wants to give us a new example to process.
void SiMaSUT::IssueQuery(const std::vector<QuerySample>& samples) {
  // Loadgen should only call this after we completed the last one.
  // So we can assume we have access to the next_example_* fields.
  // printf("IssueQuery(...) Thread=UNknown\n");
  // printf("Location of next_example_ids: %lx\n", &next_example_ids);
  {
    std::lock_guard<std::mutex> lock(LoadgenWrapperCtrl::query_mutex);

    LoadgenWrapperCtrl::next_example_ids.clear();
    LoadgenWrapperCtrl::cur_ids.clear();

    if (samples.size() > 0) {
      for (size_t i = 0; i < samples.size(); i++) {
        // fprintf(stderr, "Issuing query on: %ld, %d\n", samples[i].index, samples.size());
        LoadgenWrapperCtrl::next_example_ids.push_back(samples[i].id);
        LoadgenWrapperCtrl::cur_ids.push_back(samples[i].index);
      }

    } else {
      // Signals mlperf is done here
      std::cout << "--- MLPerf done no samples available\n";
      LoadgenWrapperCtrl::is_mlperf_done.store(1, std::memory_order_relaxed);
    }
    // std::cout << "--- MLPerf number of samples available" << samples.size() << "\n";
    LoadgenWrapperCtrl::next_example_ready = LoadgenWrapperCtrl::last_seen + 1;
  }

  LoadgenWrapperCtrl::query_cv.notify_one();
}

const std::string& SiMaSUT::Name() {
  return sut_name;
}

void SiMaSUT::FlushQueries() {
}

const std::string& SiMaQSL::Name() {
  return qsl_name;
}
