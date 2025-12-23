.. meta::
   :description: Use XGBoost with ROCm support on AMD GPUs
   :keywords: amd, rocm, finance, financial, fintech, algorithm, gpu

**************************
ROCm XGBoost documentation
**************************

XGBoost on ROCm brings GPU‑accelerated gradient boosting to AMD hardware so you
can build scalable, high‑performance machine learning applications for
financial risk modeling and other data‑intensive tasks.

XGBoost is well‑suited for financial applications due to its level‑wise tree
growth (for balanced, accurate models), strong L1 and L2 regularization against
noisy or outlier‑heavy data, and built‑in handling of missing values. Its
tunable hyperparameters achieve precision in high‑stakes tasks such as loan
default prediction, while interpretability tools, such as feature importance,
help maintain regulatory compliance. GPU acceleration can provide 3–10x
performance gains on large, feature‑rich datasets by parallelizing split
computations; however, memory demand increases with very deep trees.

ROCm enablement fully accelerates XGBoost on AMD GPUs using optimized kernels,
enhanced memory management, and seamless multi‑GPU scaling, delivering
substantial performance gains over CPU‑only baselines on typical financial
workloads.

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
