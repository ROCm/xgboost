.. meta::
   :description: Learn about XGBoost with ROCm support on AMD GPUs
   :keywords: amd, rocm, finance, xgboost, gradient boosting, gpu

.. _what-is-xgboost:

********************************************************************
What is XGBoost?
********************************************************************

XGBoost (Extreme Gradient Boosting) is a gradient boosting framework that builds
an ensemble of decision trees in sequence. Each new tree corrects errors from
previous trees, producing accurate models for classification and regression on
structured, tabular data.

On AMD hardware, XGBoost runs through the ROCm stack as GPU-accelerated
gradient boosting. The ROCm-Finance distribution targets AMD Instinct GPUs,
delivering scalable, high-performance machine learning for financial risk
modeling and other data-intensive workloads. Optimized kernels, enhanced memory
management, and multi-GPU scaling accelerate training compared to CPU-only
baselines.

How XGBoost works
====================================================================

XGBoost uses level-wise tree growth to generate balanced, accurate models.
Strong L1 and L2 regularization improve robustness against noisy data and
outliers, and missing values are handled automatically during training. Tunable
hyperparameters support the precision required for tasks such as loan default
prediction, while interpretability tools such as feature importance support
regulatory compliance.

On AMD GPUs, training parallelizes split computations across the device,
speeding up processing on large datasets. Memory requirements grow with tree depth,
so workload sizing should account for both model complexity and GPU memory.

Features and use cases
====================================================================

ROCm XGBoost on AMD hardware provides:

* **GPU-accelerated gradient boosting**: Train tree ensembles on AMD Instinct
  GPUs through the ROCm platform
* **Optimized kernels**: ROCm-tuned compute paths for split finding and tree
  construction
* **Enhanced memory management**: Efficient use of GPU memory for large
  tabular datasets
* **Multi-GPU scaling**: Distribute training across multiple AMD GPUs for
  enterprise-scale workloads
* **ROCm-Finance integration**: Install and operate alongside other ROCm-Finance
  libraries on supported ROCm versions
* **Regularization and missing-value handling**: Built-in L1/L2 penalties and
  automatic handling of missing values
* **Model interpretability**: Feature importance and related tools for
  explainable predictions in regulated environments

Why XGBoost?
====================================================================

ROCm XGBoost is commonly used where tabular machine learning must run at scale
on AMD GPUs, including:

* **Financial risk modeling**: Loan default prediction, credit scoring, and
  portfolio risk analysis on large historical datasets
* **Regulatory and compliance workflows**: Models with interpretable feature
  importance for audit and reporting requirements
* **High-throughput training**: GPU parallelization of split computations for
  faster iteration on data-intensive workloads
* **Multi-GPU enterprise deployments**: Scale training across multiple AMD
  Instinct GPUs in ROCm-Finance environments

