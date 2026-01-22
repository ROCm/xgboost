.. meta::
   :description: Use XGBoost with ROCm support on AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu

**************************
ROCm XGBoost documentation
**************************

XGBoost on ROCm provides GPU‑accelerated gradient boosting on AMD hardware,
enabling scalable, high‑performance machine learning for financial risk
modeling and data‑intensive workloads. This implementation utilizes optimized
kernels, enhanced memory management, and multi‑GPU scaling to accelerate
performance compared to CPU‑only baselines.

XGBoost excels in financial applications by using level-wise tree growth to
generate balanced, accurate models. It ensures robustness against noisy data
and outliers through strong L1 and L2 regularization and automatically manages
missing values. Tunable hyperparameters provide the high precision required for
tasks like loan default prediction, while interpretability tools such as
feature importance aid in regulatory compliance. While GPU acceleration speeds
up processing on large datasets by parallelizing split computations, memory
requirements increase with tree depth.

XGBoost is part of the `ROCm-Finance toolkit
<https://rocm.docs.amd.com/projects/rocm-finance/en/latest/>`__.

The ROCm-Finance XGBoost source code is hosted on GitHub at
`<https://github.com/ROCm/XGBoost/>`__.

ROCm-Finance XGBoost documentation is organized into the following categories:

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`/install/install`
      * :doc:`/install/build-from-source`

   .. grid-item-card:: Reference

      * `Get started with XGBoost (upstream) <https://xgboost.readthedocs.io/en/release_3.1.0/get_started.html>`__
      * `Python API (upstream) <https://xgboost.readthedocs.io/en/release_3.1.0/python/python_api.html>`__

   .. grid-item-card:: Tutorial

      * `Examples (GitHub) <https://github.com/ROCm/rocm-finance/tree/main/examples/>`__
