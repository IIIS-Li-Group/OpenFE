Tutorials on Feature Generation and FAQ
===============================================

Why Is OpenFE Ineffective For My Dataset?
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

There are two possible reasons. The first reason is that OpenFE fails to recall effective candidate features due to improper parameters. Please refer to our guidance on parameter tuning. The second reason is that there are no effective candidate features, since feature generation is not beneficial for all datasets. Our past experience on numerous real-world datasets indicates that feature generation is beneficial for 50%–70% of datasets.

How Many New Features Should I Include In The Dataset?
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

This relates to the topic of feature selection methods. Users can try to include 10, 20, 30, ... new features and see which provides the best results. It is recommended to try more delicate feature selection methods (such as forward feature selection) to achieve better performance.


