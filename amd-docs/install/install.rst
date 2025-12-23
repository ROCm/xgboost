.. meta::
   :description: Install XGBoost with ROCm support for AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu, install, setup, env, docker, pip, package, quick, start, lib

*****************************
Install ROCm XGBoost with pip
*****************************

.. _xgboost-install-prerequisites:

Prerequisites
=============

Before proceeding, ensure that you have installed a supported ROCm version,
operating system, and Python version that are compatible with the ROCm-Finance
libraries. Verify that your system includes an AMD GPU fully supported by ROCm
Finance. For guidance, see `ROCm-Finance installation prerequisites
<https://rocm.docs.amd.com/projects/rocm-finance-internal/en/latest/install/prerequisites.html>`__.

For a consistent and streamlined setup experience, it's recommended to use
a ROCm development environment Docker container. See
`Install ROCm Finance <https://rocm.docs.amd.com/projects/rocm-finance-internal/en/latest/install/install.html>`__ for instructions.

Install using pip
=================

Install the ROCm-enabled XGBoost library from the AMD-hosted PyPI repository.

.. tab-set::

   .. tab-item:: ROCm 7.0.2
      :sync: rocm7

      .. code-block:: shell

         pip install amd_xgboost --extra-index-url=https://pypi.amd.com/rocm-7.0.2/simple

   .. tab-item:: ROCm 6.4.4
      :sync: rocm6

      .. code-block:: shell

         pip install amd_xgboost --extra-index-url=https://pypi.amd.com/rocm-6.4.4/simple

Verify your installation
------------------------

Use ``pip show`` to verify your installation:

.. code-block:: shell

   pip show -v amd_xgboost

.. dropdown:: Example output

   .. code-block:: shell-session

      Name: amd_xgboost
      Version: 3.1.1
      Summary: XGBoost Python Package
      ... [output truncated]

After installing XGBoost, import and use the library. For example:

.. code-block:: python

   import xgboost as xgb
   import numpy as np
   data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
   label = np.random.randint(2, size=5)  # binary target
   dtrain = xgb.DMatrix(data, label=label)

