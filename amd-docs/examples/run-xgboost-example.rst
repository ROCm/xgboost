.. meta::
  :description: Run an example with XGBoost
  :keywords: XGBoost, ROCm, example, sample, tutorial

.. _run-xgboost-example:

********************************************************************
Run an XGBoost example
********************************************************************

For comprehensive tutorials and guides on using XGBoost, refer to the upstream XGBoost documentation:

* `XGBoost Documentation <https://xgboost.readthedocs.io/en/release_3.2.0/>`__

* `Getting Started Guide <https://xgboost.readthedocs.io/en/stable/get_started.html>`__

* `XGBoost 3.2.0 Release Notes <https://xgboost.readthedocs.io/en/release_3.2.0/changes/v3.2.0.html>`__

For detailed Python API documentation, refer to:

* `Python API Guide <https://xgboost.readthedocs.io/en/latest/python/python_api.html>`__

* `Python Introduction <https://xgboost.readthedocs.io/en/latest/python/python_intro.html>`__

Key Python modules
====================================================================

**Core Training:**

* ``xgboost.train()``: Core training function

* ``xgboost.DMatrix``: Data matrix for XGBoost

* ``xgboost.Booster``: Model object

**Scikit-learn Interface:**

* ``xgboost.XGBClassifier``: Classifier compatible with scikit-learn

* ``xgboost.XGBRegressor``: Regressor compatible with scikit-learn

* ``xgboost.XGBRanker``: Ranking model

**Distributed Training:**

* ``xgboost.dask``: Dask integration for distributed training

* ``xgboost.spark``: Spark integration

Quick example
====================================================================

.. code-block:: python

   import xgboost as xgb
   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   
   # Load data
   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Create DMatrix
   dtrain = xgb.DMatrix(X_train, label=y_train)
   dtest = xgb.DMatrix(X_test, label=y_test)
   
   # Set parameters for AMD GPU
   params = {
       'tree_method': 'hist',
       'device': 'cuda',
       'max_depth': 6,
       'eta': 0.3,
       'objective': 'binary:logistic',
       'eval_metric': 'logloss'
   }
   
   # Train
   model = xgb.train(params, dtrain, num_boost_round=100,
                     evals=[(dtest, 'test')], early_stopping_rounds=10)
   
   # Predict
   predictions = model.predict(dtest)


Examples repository
====================================================================

The `https://github.com/ROCm/rocm-finance/tree/release/26.01/examples/xgboost <https://github.com/ROCm/rocm-finance/tree/release/26.01/examples/xgboost>`__ repository contains comprehensive examples demonstrating GPU-accelerated machine learning using AMD ROCm.

The ``xgboost`` directory includes:

* ``README.md``: Documentation providing setup instructions, requirements, and usage guidelines for the XGBoost fraud detection demo
* ``xgboost_example.py``: The primary application file implementing an interactive fraud detection system

XGBoost Credit Card Fraud Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This demonstration showcases accelerated credit card fraud detection using AMD ROCm GPU and XGBoost. Key features include the following:

- **GPU training:** Leverages ROCm-enabled AMD graphics processors for model training
- **Performance comparison:** Enables optional CPU benchmarking to measure GPU speedup
- **Model evaluation:** Generates accuracy metrics, AUC scores, and confusion matrices
- **Data visualization:** Displays top 20 suspicious transactions and prediction analysis
- **Export functionality:** Saves predictions to CSV format
- **Interactive interface:** Uses Gradio framework for web-based interaction

The example demonstrates how to train XGBoost on ROCm-enabled AMD GPUs while providing optional CPU training for speed comparison. Users can configure hyperparameters and observe performance differences between GPU and CPU implementations on the credit card fraud dataset.

Run the example
--------------------------------------------------------------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ROCm/rocm-finance.git
   cd rocm-finance/examples/xgboost
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the fraud detection demo
   python xgboost_example.py


For more examples and use cases, see the `XGBoost Demo Gallery <https://github.com/ROCm/xgboost/tree/release/3.2.0/demo>`__.
