.. meta::
   :description: Build XGBoost with ROCm support from source for AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu, install, setup, env, docker, package, contribute, develop, build, pip, make

.. _build-xgboost-source:

******************************
Build XGBoost from source
******************************

Prerequisites
=============

To use XGBoost `3.2.0 <https://github.com/dmlc/xgboost/tree/release_3.2.0>`__ on ROCm, you need the following prerequisites:

- **ROCm version:** `7.2.3 <https://rocm.docs.amd.com/en/docs-7.2.3/>`__, `7.1.1 <https://rocm.docs.amd.com/en/docs-7.1.1/>`__, `7.0.2 <https://rocm.docs.amd.com/en/docs-7.0.2/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI355X (ROCm 7.2.3 only), MI325X, MI300X
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31213/>`__, `3.11 <https://www.python.org/downloads/release/python-31115/>`__, `3.10 <https://www.python.org/downloads/release/python-31020/>`__

For a consistent and streamlined setup experience, it's recommended to use
a ROCm development environment Docker container. See `Install AMD Finance
<https://rocm.docs.amd.com/projects/rocm-finance-internal/en/docs-26.05/install/install.html>`__
for instructions.

Build from source
=================

1. Install required software dependencies.

   .. code-block:: shell

      apt-get update && apt-get install -y --no-install-recommends git libomp-dev python3-venv curl ca-certificates gpg wget
      wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
      echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
      apt-get update
      apt-get install -y cmake libgtest-dev libgmock-dev

2. Clone the `<https://github.com/AMD-Ecosystem/xgboost>`__ source code from GitHub.

   .. code-block:: shell

      git clone --recurse-submodules https://github.com/AMD-Ecosystem/xgboost.git

3. Create and activate a Python virtual environment.

   .. code-block:: shell

      python3 -m venv xgboost-build
      source xgboost-build/bin/activate

4. Build the shared object library.

   .. code-block:: shell

      # Set target GPU architecture
      export AMDGPU_TARGETS="gfx942"  # MI300X/MI325
      # For MI350/MI355, use gfx950

      export CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake/
      export CMAKE_MODULE_PATH=/opt/rocm/lib/cmake
      export CMAKE_C_COMPILER=hipcc
      export CMAKE_CXX_COMPILER=hipcc
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/llvm/lib/
      export LDFLAGS="-L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib"

      cd xgboost

      # Configure with CMake
      cmake -DUSE_HIP=1 -DUSE_RCCL=1 -S . -B build

      # Build (adjust -j based on your CPU cores)
      cmake --build build -- -j 8

   Build Options:

   - ``-DUSE_HIP=1``: Enable ROCm/HIP support
   - ``-DUSE_RCCL=1``: Enable RCCL for multi-GPU support

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

      .. tab-item:: ROCm 7.2.3

         .. code-block:: shell

            pip install pytest joblib hypothesis scikit-learn rich fastrlock cachetools fsspec packaging \
              numba==0.60 pandas==2.2.3 pyarrow==19.0 typing-extensions>=4.0.0
            pip install amd-cupy amd-hipdf --index-url=https://pypi.amd.com/rocm-7.2.3/simple/

      .. tab-item:: ROCm 7.1.1

         .. code-block:: shell

            pip install pytest joblib hypothesis scikit-learn rich fastrlock cachetools fsspec packaging \
              numba==0.60 pandas==2.2.3 pyarrow==19.0 typing-extensions>=4.0.0
            pip install amd-cupy amd-hipdf --index-url=https://pypi.amd.com/rocm-7.1.1/simple/

      .. tab-item:: ROCm 7.0.2

         .. code-block:: shell

            pip install pytest joblib hypothesis scikit-learn rich fastrlock cachetools fsspec packaging \
              numba==0.60 pandas==2.2.3 pyarrow==19.0 typing-extensions>=4.0.0
            pip install amd-cupy amd-hipdf --index-url=https://pypi.amd.com/rocm-7.0.2/simple/

8. Run the Python examples.

   .. code-block:: shell

      export ROCM_HOME=/opt/rocm
      pytest tests/python-gpu
