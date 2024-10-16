#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
//#include "thread_bind.h"
#include "kmp_launcher.hpp"

namespace py = pybind11;

namespace binder {

  void bind_thread(int startIndex, int len) {
    kmp::KMPLauncher thCtrl;
    std::vector<int> places(len);
    for (int i = 0; i < len; ++i) {
        places[i] = i + startIndex;
    }
    thCtrl.setAffinityPlaces(places).pinThreads();
  }

}


PYBIND11_MODULE(thread_binder, m){
  m.doc() = "Core binder for python process.";
  m.def("bind_thread", &binder::bind_thread);
}