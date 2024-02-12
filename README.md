## CUDA Kernels profiling
NVCC compiler flags for building the building the kernels in VSCode without the help of `torch.utils.cpp_extension.load_inline`
```JSON
    "-L../path_to_conda_env/lib/python3.11/site-packages/torch/lib",
    "-I../path_to_conda_env/lib/python3.11/site-packages/torch/include",
    "-I../path_to_conda_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include",
    "-I../path_to_conda_env/lib/python3.11/site-packages/torch/include/TH",
    "-I../path_to_conda_env/lib/python3.11/site-packages/torch/include/THC",
    "-I../path_to_conda_env/include/python3.11",
    "--expt-relaxed-constexpr",
    "-D_GLIBCXX_USE_CXX11_ABI=0",
    "-lc10",
    "-ltorch_cpu",
    "-ltorch",
    "-ltorch_python",
```
`includePath`
```JSON
    "includePath": [
        "${workspaceFolder}/**",
        "${workspaceFolder}/../../cpp/libtorch-cxx11-abi-shared-with-deps-2.2.0+cu121/**",
        "$..path_to_dir/miniconda3/envs/torch/include/python3.11/**",
        "..path_to_dir/libtorch-cxx11-abi-shared-with-deps-2.2.0+cu121/libtorch/include/torch/csrc/api/include"
]
```