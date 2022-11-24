Quick Start
============

**Installation**\ :

It is recommended to use **pip** for installation.

.. code-block:: bash

   pip install openfe            # normal install
   pip install --upgrade openfe  # or update if needed

**A Quick Example**\ :

.. code-block:: bash

    from openfe import openfe, transform

    ofe = openfe()
    features = ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs) # transform the train and test data according to generated features.

We provide an example using the standard california_housing dataset in `this link <https://github.com/ZhangTP1996/OpenFE/blob/master/examples/california_housing.py>`_. A more complicated example demonstrating OpenFE can outperform machine learning experts in the IEEE-CIS Fraud Detection Kaggle competition is provided in `this link <https://github.com/ZhangTP1996/OpenFE/blob/master/examples/IEEE-CIS-Fraud-Detection/main.py>`_. 

**Required Dependencies**\ :

* Python>=3.8
* numpy>=1.22.3
* pandas>=1.4.1
* scikit-learn>=1.0.2
* lightgbm>=3.3.2
* scipy>=1.9.1
* tqdm



