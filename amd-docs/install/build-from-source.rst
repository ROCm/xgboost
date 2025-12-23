.. meta::
   :description: Build XGBoost with ROCm support from source for AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu, install, setup, env, docker, package, contribute, develop, build, pip, make

******************************
Build ROCm XGBoost from source
******************************

Prerequisites
=============

Before proceeding, ensure that you have installed a supported ROCm version,
operating system, and Python environment that are compatible with the
ROCm-Finance libraries. Verify that your system includes a supported AMD
Instinct GPU. For guidance, see `ROCm-Finance installation prerequisites
<https://rocm.docs.amd.com/projects/rocm-finance/en/latest/install/prerequisites.html>`__.

For a consistent and streamlined setup experience, it's recommended to use
a ROCm development environment Docker container. See `Install ROCm-Finance
<https://rocm.docs.amd.com/projects/rocm-finance/en/latest/install/install.html>`__
for instructions.

Build from source
=================

1. Install required software dependencies.

   .. code-block:: shell

      RUN apt-get update && apt-get install -y --no-install-recommends git libomp-dev python3-venv curl ca-certificates gpg wget
      RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
      RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
      RUN apt-get update
      RUN apt-get install -y cmake libgtest-dev libgmock-dev

2. Clone the `<https://github.com/ROCm/xgboost>`__ source code from GitHub.

   .. code-block:: shell

      git clone --recurse-submodules https://github.com/rocm/xgboost.git

3. Create and activate a Python virtual environment.

   .. code-block:: shell

      python -m venv xgboost-build
      source xgboost-build/bin/activate

4. Build the shared object library.

   .. code-block:: shell

      # Set the GPU target
      export AMDGPU_TARGETS="gfx942"
      export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake/
      export CMAKE_MODULE_PATH=/opt/rocm/lib/cmake
      export CMAKE_POLICY_VERSION_MINIMUM=3.5
      export CMAKE_C_COMPILER=hipcc
      export CMAKE_CXX_COMPILER=hipcc
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/llvm/lib/
      export LDFLAGS="-L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib"

      cd xgboost
      cmake -DUSE_HIP=1 -DUSE_RCCL=1 -DGOOGLE_TEST=1 -S . -B build
      cmake --build build -- -j 8

5. Build and install the Python package.

   .. code-block:: shell

      cd python-package
      pip install setuptools pytest build wheel
      python3 -m build --wheel .
      pip install dist/amd_xgboost*.whl 

6. Verify the installation.

   .. code-block:: shell

      pip show amd_xgboost

7. Install dependencies.

   .. tab-set::

      .. tab-item:: ROCm 7.0.2

         .. code-block:: shell

            pip install pytest joblib hypothesis scikit-learn rich fastrlock cachetools fsspec packaging pip install numba==0.60 pip install pandas==2.2.3 pip install pyarrow==19.0 pip install typing-extensions>=4.0.0 pip install amd-cupy --index-url=https://pypi.amd.com/rocm-7.0.2/simple/ pip install amd-hipdf --index-url=https://pypi.amd.com/rocm-7.0.2/simple/

      .. tab-item:: ROCm 6.4.4

         .. code-block:: shell

            pip install pytest joblib hypothesis scikit-learn rich fastrlock cachetools fsspec packaging pip install amd-cupy amd-hipdf --extra-index-url=https://pypi.amd.com/simple/

8. Run the Python examples.

   .. code-block:: shell

      pytest tests/python-gpu
