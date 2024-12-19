# ROCm version

ROCm 5.5 and newer

# Code
Clone the code from our repo

1. `git clone https://github.com/ROCmSoftwarePlatform/xgboost`
1. `cd xgboost`
1. `git checkout master-rocm`

or a tag/branch with rocm suffix, such as v2.0.1-rocm

# Submodules
XGBoost ROCm support requires a few modules, which can be initialized as,

`git submodule update --init --recursive`

# Configure
The following export may be required for some systems, and the ROCm path depends on installation,

1. `export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/rocm/lib/cmake:/opt/rocm/lib/cmake/AMDDeviceLibs/`
1. `mkdir build`
1. `cd build`
1. `cmake -DUSE_HIP=ON ../`
1. or `cmake -DUSE_HIP=1 ../`
1. or `cmake -DUSE_HIP=1 -DUSE_RCCL=1 ../`
1. or `cmake -DUSE_HIP=1 -DGOOGLE_TEST=1 ../`

The first command may be optional depending on system configure.

The **USE_HIP** macro enables HIP/ROCm support. **USE_RCCL** enables RCCL. **GOOGLE_TEST** enables Google test.

apt-get install libgtest-dev libgmock-dev

# Compile
To compile, run command,

`make -j`

# Python Support
After compilation, XGBoost can be installed as a Python package and supports a wide range of applications,

1. `cd python-package/`
1. `pip3 install .`

# Use AMD GPUs
When calling XGBoost, set the parameter `device` to `gpu` or `cuda`. Python sample,

```
params = dict()
params["device"] = "gpu"
params["tree_method"] = "hist"
...
```

or

```
params = dict()
params["device"] = "cuda"
params["tree_method"] = "hist"
...
```

# R Installation

1. clone code
1. update submodules
1. mkdir build; cd build
1. cmake .. -DUSE_HIP=ON -DR_LIB=ON
1. make -j && make install
1. cd R-package-install/R-package/
1. R CMD INSTALL .
