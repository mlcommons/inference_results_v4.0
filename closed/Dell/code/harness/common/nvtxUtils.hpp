/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

// ================================
//     Debug support: nvtx ranges
// ================================
#ifndef NVTX_UTILS_HPP
#define NVTX_UTILS_HPP

#include <nvtx3/nvToolsExt.h> // For NVTX annotations
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <queue>
#include <string>
#include <thread>

namespace nvtx
{
#define RGB_COLOR(R, G, B) ((R << 16) + (G << 8) + B)
enum nvtx_color
{
    COLOR_BLUE_0 = 255,
    COLOR_BLUE_1 = 200,
    COLOR_BLUE_2 = 150,
    COLOR_BLUE_3 = 100,
    COLOR_GREEN_0 = 255 << 8,
    COLOR_GREEN_1 = 200 << 8,
    COLOR_GREEN_2 = 150 << 8,
    COLOR_GREEN_3 = 100 << 8,
    COLOR_RED_0 = 255 << 16,
    COLOR_RED_1 = 200 << 16,
    COLOR_RED_2 = 150 << 16,
    COLOR_RED_3 = 100 << 16,
    COLOR_YELLOW_0 = RGB_COLOR(255, 255, 0),
    COLOR_YELLOW_1 = RGB_COLOR(189, 183, 107),
    COLOR_YELLOW_2 = RGB_COLOR(238, 232, 170),
    COLOR_YELLOW_3 = RGB_COLOR(255, 250, 205),
    COLOR_YELLOW_4 = RGB_COLOR(240, 230, 140),
    COLOR_YELLOW_5 = RGB_COLOR(255, 255, 224),
    COLOR_YELLOW_6 = RGB_COLOR(255, 228, 181),
    COLOR_YELLOW_7 = RGB_COLOR(255, 218, 185),
    COLOR_PINK_0 = RGB_COLOR(255, 192, 203),
    COLOR_PINK_1 = RGB_COLOR(255 - 30, 192 - 30, 203 - 30)
};

inline nvtxRangeId_t global_event_start(std::string const& eventName, nvtx_color const eventColor, int64_t payload = 0)
{
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = eventColor;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = eventName.data();
    eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    eventAttrib.payload.ullValue = payload;

    nvtxRangeId_t corrId = nvtxRangeStartEx(&eventAttrib);
    return (corrId);
}

inline void global_event_end(nvtxRangeId_t corrId)
{
    nvtxRangeEnd(corrId);
}

inline void thread_event_start(std::string const& eventName, nvtx_color const eventColor)
{
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = eventColor;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = eventName.data();
    nvtxRangeId_t corrId = nvtxRangePushEx(&eventAttrib);
}

inline void thread_event_end()
{
    nvtxRangePop();
}

inline void mark_event(std::string const& eventName, nvtx_color const eventColor)
{
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = eventColor;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = eventName.data();
    nvtxMarkEx(&eventAttrib);
}

} // namespace nvtx

#define NVTX_GLOBAL_START(A, B, C) nvtxRangeId_t A = nvtx::global_event_start(B, C)
#define NVTX_GLOBAL_START_WITH_PAYLOAD(A, B, C, D) A = nvtx::global_event_start(B, C, D)
#define NVTX_GLOBAL_END(A) nvtx::global_event_end(A)
#define NVTX_THREAD_START(B, C) nvtx::thread_event_start(B, C)
#define NVTX_THREAD_END() nvtx::thread_event_end()
#define NVTX_MARK(B, C) nvtx::mark_event(B, C)


// ----------------
// Perfetto tracing
// ----------------

#if 1
// PerfTrPrinter
//

class PerfTrPrinter {
private:
   std::queue<std::string> mBuffer;
   std::mutex mMtx;
   std::condition_variable mCv;
   std::atomic<bool> mDone{false};
   std::thread printerThread;

   void printLoop() {
       while (!mDone || !mBuffer.empty()) {
           std::unique_lock<std::mutex> lock(mMtx);
           mCv.wait(lock, [this] { return !mBuffer.empty() || mDone; });
           if (!mBuffer.empty()) {
               std::cout << mBuffer.front() << std::endl;
               mBuffer.pop();
           }
       }
   }

public:
   PerfTrPrinter() : printerThread(&PerfTrPrinter::printLoop, this) {}

   ~PerfTrPrinter() {
       mDone = true;
       mCv.notify_one();
       printerThread.join();
   }

   void print(const std::string& str) {
       std::lock_guard<std::mutex> guard(mMtx);
       mBuffer.push(str);
       mCv.notify_one();
   }
};
#endif

#if 1
namespace perfTr
{

enum PtRecord
{
   kPUSH = 0,
   kPOP = 1,
   kEVENT = 2
};

enum PtField
{
    kUNDEF = 0,
    kBS = 1,
    kISL = 2,
    kOSL = 3
};

enum PtTag
{
    kENQ = 0,      // enqueue event
    kDEQ = 1,      // dequeue event
    kBATCH = 2     // batching event
};

inline int64_t perfTrTime() {
    auto start = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count();
    return milliseconds;
}

inline void perfTrRecord(PtRecord record, PtTag tag, PtField field, uint64_t value, PerfTrPrinter &pr)
{
    pthread_t threadId = pthread_self();
    auto time = perfTrTime();

    // PTR;thread;time;record;tag;count:field:value:...
    std::stringstream ss;
    ss << "PTR:" << threadId 
       << ";" << time 
       << ";" << record 
       << ";" << tag 
       << ";" << 1 
       << ";" << field 
       << ";" << value;

    // std::cout << ss.str();
    pr.print(ss.str());
}

inline void perfTrPush(PtTag tag, PtField field, uint64_t value, PerfTrPrinter &pr) {
    perfTrRecord(kPUSH, tag, field, value, pr);
}

inline void perfTrPop(PtTag tag, PtField field, uint64_t value, PerfTrPrinter &pr) {
    perfTrRecord(kPOP, tag, field, value, pr);
}

inline void perfTrEvent(PtTag tag, PtField field, uint64_t value, PerfTrPrinter &pr) {
    perfTrRecord(kEVENT, tag, field, value, pr);
}


} // namespace perfTr

#define PERF_TR_PUSH(A, B, C, D) perfTr::perfTrPush(A, B, C, D)
#define PERF_TR_POP(A, B, C, D) perfTr::perfTrPop(A, B, C, D)
#define PERF_TR_EVENT(A, B, C, D) perfTr::perfTrEvent(A, B, C, D)

// sub-sampled event macro: accumulate updates using SUBSAMP rate with value UPDATE
#define PERT_TR_EVENT_SUBSAMPLE(SUBSAMP, UPDATE, A, B, D)         \
    static int sCnt =0 ;                                          \
    static int sAcc =0 ;                                          \
                                                                  \
    sCnt++;                                                       \
    sAcc += UPDATE;                                               \
    if (sCnt % SUBSAMP == 0) {                                    \
        perfTr::perfTrEvent(A, B, sAcc, D);                       \
        sAcc = 0;                                                 \
    }                                                                        

#endif

#endif // NVTX_UTILS_HPP
