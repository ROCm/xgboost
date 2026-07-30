.. meta::
   :description: Install XGBoost with ROCm support for AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu, install, setup, env, docker, pip, package, quick, start, lib

.. _install-xgboost-pip:

*****************************
Install XGBoost with pip
*****************************

Prerequisites
=============

To use XGBoost `3.2.0 <https://github.com/dmlc/xgboost/tree/release_3.2.0>`__ on ROCm, you need the following prerequisites:

- **ROCm version:** `7.2.3 <https://rocm.docs.amd.com/en/docs-7.2.3/>`__, `7.1.1 <https://rocm.docs.amd.com/en/docs-7.1.1/>`__, `7.0.2 <https://rocm.docs.amd.com/en/docs-7.0.2/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI355X (ROCm 7.2.3 only), MI325X, MI300X
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31213/>`__, `3.11 <https://www.python.org/downloads/release/python-31115/>`__, `3.10 <https://www.python.org/downloads/release/python-31020/>`__

For a consistent and streamlined setup experience, it's recommended to use
a ROCm development environment Docker container. See
`Install AMD Finance <https://rocm.docs.amd.com/projects/rocm-finance/en/docs-26.05/install/install.html>`__ for instructions.

Install using pip
=================

Install the ROCm-enabled XGBoost library from the AMD-hosted PyPI (Python Package Index) repository.

.. tab-set::

   .. tab-item:: ROCm 7.2.3
      :sync: rocm723

      .. code-block:: shell

         pip install amd_xgboost --extra-index-url=https://pypi.amd.com/rocm-7.2.3/simple

   .. tab-item:: ROCm 7.1.1
      :sync: rocm711

      .. code-block:: shell

         pip install amd_xgboost --extra-index-url=https://pypi.amd.com/rocm-7.1.1/simple

   .. tab-item:: ROCm 7.0.2
      :sync: rocm702

      .. code-block:: shell

         pip install amd_xgboost --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple

Verify your installation
------------------------

Use ``pip show`` to verify your installation:

.. code-block:: shell

   pip show -v amd_xgboost

.. dropdown:: Example output

   .. code-block:: shell-session

      Name: amd_xgboost
      Version: 3.2.0
      Summary: XGBoost Python Package
      ... [output truncated]

After installing XGBoost, import and use the library. For example:

.. code-block:: python

   import xgboost as xgb
   print(f"XGBoost version: {xgb.__version__}")

   # Create sample data
   from sklearn.datasets import make_classification
   X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

   # Test GPU training
   dtrain = xgb.DMatrix(X, label=y)
   params = {'tree_method': 'hist', 'device': 'cuda', 'max_depth': 3}
   model = xgb.train(params, dtrain, num_boost_round=10)
   print("GPU training successful!")
