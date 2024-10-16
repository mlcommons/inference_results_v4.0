#pragma once

#include <sys/sysinfo.h>
#include <unistd.h>

#include <ATen/core/ivalue.h>
#include <string>
#include <future>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "bert_qsl.hpp"
#include "kmp_launcher.hpp"

namespace models {
//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
public:
  TorchModel (const std::string filename, int max_numa) {
    auto load_model = [filename](int socket, int max_numa){ 
      kmp::KMPAffinityMask mask;
      int curMaxProc = std::thread::hardware_concurrency();
      mask.addCore(curMaxProc - mask.cores_per_nnode(max_numa) * socket - 1).bind();  

      auto model = torch::jit::load(filename);
      // sched_setaffinity(0, sizeof(backup), &backup);
      return model;
    };

    for(int i = 0; i < max_numa && i < MAX_NUMA_NODE; i++){
       std::future<torch::jit::script::Module> m = std::async(std::launch::async, load_model, i, max_numa);
       socket_model_[i] = m.get();
    }
  }

  TorchModel ();

  // void load(const std::string filename) {
  //   model_ = torch::jit::load(filename);
  // }

  template <typename... Args>
  at::IValue inference(Args&&... args) {
    return socket_model_[0].forward(std::forward<Args>(args)...);
  }

  template <typename... Args>
  at::IValue inference_at(int socket, Args&&... args) {
    return socket_model_[socket].forward(std::forward<Args>(args)...);
  }

private:
  torch::jit::script::Module socket_model_[MAX_NUMA_NODE];
};

}
