Quick Start
============

**Installation**\ :

It is recommended to use **pip** for installation.

.. code-block:: bash

   pip install openfe            # normal install
   pip install --upgrade openfe  # or update if needed

Please do not use **conda install openfe** for installation.
It will install another python package different from ours.

**A Quick Example**\ :

.. code-block:: bash

    from openfe import OpenFE, transform

    ofe = OpenFE()
    features = ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)  # generate new features
    train_x, test_x = transform(train_x, test_x, features, n_jobs=n_jobs) # transform the train and test data according to generated features.

We provide an `example <https://github.com/IIIS-Li-Group/OpenFE/blob/master/examples/california_housing.py>`_ using the standard california_housing dataset.
A more complicated `example <https://github.com/IIIS-Li-Group/OpenFE/blob/master/examples/IEEE-CIS-Fraud-Detection/>`_ demonstrating OpenFE can outperform machine learning experts in the IEEE-CIS Fraud Detection Kaggle competition.
Users can also refer to `our paper <https://arxiv.org/abs/2211.12507>`_ for more details of OpenFE.

**Required Dependencies**\ :

* Python>=3.6
* numpy>=1.19.3
* pandas>=1.1.5
* scikit-learn>=0.24.2
* lightgbm>=3.3.2
* scipy>=1.5.4
* xgboost>=1.5.2
* tqdm
* pyarrow



