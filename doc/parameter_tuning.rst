Parameters Tuning
=================

This page contains parameters tuning guides for different scenarios.


For Faster Speed
----------------

Add More Computational Resources
''''''''''''''''''''''''''''''''

On systems where it is available, OpenFE uses multiprocessing to parallelize the calculation and evaluation of new features. The maximum number of threads used by OpenFE is controlled by the parameter ``n_jobs``. For best performance, set this to the number of **real** CPU cores available.

Increase ``n_data_blocks``
''''''''''''''''''''''''''''''''''''''''''''''''

This parameter is an integer that controls the number of data blocks the data is split into for successive feature-wise halving. The overall complexity of OpenFE is :math:`O(q2^{-q}\cdot mn^2)`, where :math:`m` is the number of features, :math:`n` is the number of samples, and :math:`2^q` is the number of data blocks. However, setting ``n_data_blocks`` to a large value can achieve faster speed, but may hurt the overall performance since useful candidate features may be discarded during successive feature-wise halving. This is a trade-off between efficiency and effectiveness. Besides, setting ``n_data_blocks`` to a large value will result in few candidate features left for stage2 selection. This can be controlled by setting the parameter ``min_candidate_features`` to early-stop the successive feature-wise halving.

Perform Feature Selection Before Feature Generation
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Another way to speed up is to perform feature selection and remove redundant features before feature generation. This can reduce the number of candidate features to achieve speed up. However, this way of speeding up may also hurt the performance, because uninformative features may also yield informative candidate features after transformation. In a Diabetes dataset, for example, when the goal is to forecast if a patient will be readmitted to the hospital, the feature 'patient id' is useless. However, 'freq(patient id),' which is the number of times the patient has been admitted to the hospital, is a strong predictor of whether the patient would be readmitted. An alternative way to  is to determine the ``candidate_features_list`` yourself by ``openfe.get_candidate_features()``. 

In a word, there is no free speeding up method. If there is, we would have been glad to implement it in our algorithm.

For Better Performance
-----------------------------

Decrease ``n_data_blocks`` Or Increase ``min_candidate_features``
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

This is particularly important if there are many useful candidate features in the pool of candidate features. However, changing these two parameters may not make a difference if there isn't many useful candidate features.

Perform Feature Selection on Generated Features
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

In our examples, we directly include the top generated features recommended by OpenFE. Users can also perform more delicate feature selection methods (such as forward feature selection) to achieve better performance.

Set ``feature_boosting`` to True or False and See Which Provides Better Results
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

``feature_boosting`` is an important parameter that determines whether to use feature boosting in OpenFE (see more details in our paper). In general, using feature boosting yields better results on most datasets. However,  on some of the datasets, we find that disabling feature boosting can provide better results. We are planning to investigate this issue further in the future.