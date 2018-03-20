# FNC-Project-Stance

The project is based upon https://github.com/FakeNewsChallenge/fnc-1-baseline

# How to run?
    You can simply run the project by `pyhton3 main.py`
    
We have cached features generation in the `features` folder. If you would like to modify any of these features, make sure you remove the related cached files before perfroming any trainings. For instances, you will have to remove all tf-idf cahces files (tf_idf.*.npy) if you change anything on the `tf_idf_features` in the `ext_feature_eng.py` file.

# Evaluation:
##### Please check the following files for analytics:
* `agree_evaluation.txt`
* `disagree_evaluation.txt`
* `discuss_evaluation.txt`
* `unrelated_evaluation.txt`
* `confusion_matrix_dev.txt`
* `confusion_matrix_test.txt`
* `svd.png`
* `tfidf.png`
