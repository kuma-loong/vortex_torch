#include "register.h"



PYBIND11_MODULE(vortex_torch_C, m){
        m.def("sglang_plan_decode",             &sglang_plan_decode);
        m.def("sglang_plan_prefill",            &sglang_plan_prefill);
        m.def("Chunkwise_NH2HN_Transpose",      &Chunkwise_NH2HN_Transpose);
        m.def("Chunkwise_HN2NH_Transpose",      &Chunkwise_HN2NH_Transpose);
        m.def("topk_output",                    &topk_output);
}
