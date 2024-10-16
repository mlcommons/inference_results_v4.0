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

#ifndef LOADGEN_WRAPPER_CTRL_H_
#define LOADGEN_WRAPPER_CTRL_H_

#include <cstdint>
#include <condition_variable>
#include <future>
#include <mutex>

#include <loadgen.h>

struct LoadgenWrapperCtrl {
  // Mutex used to protect the access the data received from IssueQuery back from loadgen
  static std::mutex query_mutex;

  // Conditional variable to let loadgen know that data is available and ready from the laodgen
  static std::condition_variable query_cv;

  // Vector of responses back to loadgen library
  static std::vector<mlperf::ResponseId> next_example_ids;

  // The last seen variable used to keep track of last seen sample ids
  static size_t last_seen;

  static size_t next_example_ready;

  // Size of query samples ids returned by Loadgen
  static size_t next_example_size;

  // A copy of the next_exaple_ids used by the system to manipulate
  static std::vector<int64_t> cur_ids;

  // Atomic variable to indicate StartTest returned from loadgen
  static std::atomic<int> is_mlperf_done;

  // The future object returned by async StartTest thread
  static std::future<int> future;
};

#endif // LOADGEN_WRAPPER_CTRL_H_
