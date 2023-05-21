import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import sklearn
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import traceback
import os
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from .FeatureGenerator import *
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
import scipy.special
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


class ForwardFeatureSelector:
    def __init__(
        self,
        estimator=None,
        features_for_selection=None,
        feature_importance='default',
        threshold=0.001,
        metric=None,
        task=None,
        step=1,
        n_jobs=4,
        verbose=True,
    ):
        """ Forward Feature Selection

        Parameters
        ----------
        estimator: object, optional (default=None)
            A scikit-learn estimator for regression or classification.
            If not passed in, LightGBM with default parameters is used.

        features_for_selection: list, optional (default=None)
            List of features to consider for selection. If not provided, all features in the dataset will be considered.

        feature_importance: str or list, optional (default='default')
            Method for calculating feature importance. Can be 'default', 'permutation', or a list of feature importances.
            'default': uses estimator's feature_importances_ attribute.
            'permutation': uses permutation_importance to calculate feature importances.
            If a list is provided, it must have the same length as the number of features in the dataset.
            Feature importance is greater the better.

        threshold: float, optional (default=0.001)
            Minimum improvement in performance required to add a new feature to the selected_features list.

        metric: str, optional (default=None)
            Scikit-learn scoring metric to use. Must be a valid scorer from sklearn.metrics.SCORERS.
            If not provided, the metric will be determined based on the task type.

        task: str, optional (default=None)
            The type of machine learning task, either 'classification' or 'regression'.
            If not provided, the task type will be inferred based on the label column.

        step: int, optional (default=1)
            Number of features to consider at each iteration during the forward selection process.

        n_jobs: int, optional (default=4)
            Number of cores to run in parallel for both the estimator and the permutation importance calculation.

        verbose: bool, optional (default=True)
            Whether to print information.

        """
        self.estimator = estimator
        self.features_for_selection = features_for_selection
        self.feature_importance = feature_importance
        self.threshold = threshold
        self.metric = metric
        self.task = task
        self.step = step
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.selected_features = []
        self.history = {'scores': [], 'selected_features': [],
                        'n_selected': [], 'n_examined': [],
                        'features_improvement': []}
        self.data = None
        self.label = None

    def fit(self, data, label):
        self.data = data
        self.label = label
        self.data_to_dataframe()

        if self.features_for_selection is None:
            self.features_for_selection = self.data.columns.to_list()
        elif set(self.features_for_selection).issubset(set(data.columns)) is False:
            raise ValueError("features_for_selection should be a subset of data.columns")

        self.get_task()
        self.get_metric()
        self.get_estimator()

        if isinstance(self.feature_importance, str):
            estimator_all = self.estimator.fit(data, label.values.ravel())
            if self.feature_importance == 'default':
                self.feature_importance = estimator_all.feature_importances_
            else:
                self.feature_importance = permutation_importance(
                    estimator_all, data, label.values.ravel(), n_jobs=self.n_jobs
                ).importances_mean
        else:
            assert isinstance(self.feature_importance, list)
            assert len(self.feature_importance) == len(self.data.columns)
        feature_importance_map = {col: imp for col, imp in zip(data.columns, self.feature_importance)}
        self.features_for_selection.sort(key=lambda x: feature_importance_map[x], reverse=True)
        self.selected_features = list(set(data.columns) - set(self.features_for_selection))
        total = len(self.features_for_selection)
        if len(self.selected_features) == 0:
            self.selected_features.append(self.features_for_selection[0])
            self.features_for_selection = self.features_for_selection[1:]
            previous_score = np.mean(cross_val_score(
                self.estimator, self.data[self.selected_features].values.reshape(-1, 1), self.label.values.ravel(), scoring=self.metric, cv=5
            ))
            total_count = 1
        else:
            total_count = 0
            previous_score = np.mean(cross_val_score(
                self.estimator, self.data[self.selected_features], self.label.values.ravel(), scoring=self.metric, cv=5
            ))
        self.history['scores'].append(previous_score)
        self.history['selected_features'].append(deepcopy(self.selected_features))
        self.history['n_selected'].append(len(self.selected_features))
        self.history['n_examined'].append(len(self.selected_features))
        while self.features_for_selection:
            count = 0
            while count < self.step and self.features_for_selection:
                feature = self.features_for_selection.pop(0)
                self.selected_features.append(feature)
                count += 1
            current_score = np.mean(cross_val_score(
                self.estimator, self.data[self.selected_features], self.label.values.ravel(), scoring=self.metric, cv=5
            ))
            improvement = current_score - previous_score
            total_count += count
            self.history['n_examined'].append(self.history['n_examined'][-1] + count)
            self.history['features_improvement'].append([self.selected_features[-count:], improvement])
            if improvement < self.threshold:
                self.my_print("Progress: [%d/%d] = %d%% "
                              "%s not selected with improvement %.4lf" %
                              (total_count, total, int(total_count / total * 100),
                               self.selected_features[-count:], improvement)
                              )
                self.selected_features = self.selected_features[:-count]
                self.history['selected_features'].append([])
            else:
                self.my_print("Progress: [%d/%d] = %d%% "
                              "%s selected with improvement %.4lf" %
                              (total_count, total, int(total_count / total * 100),
                               self.selected_features[-count:], improvement)
                              )
                previous_score = current_score
                self.history['selected_features'].append(self.selected_features[-count:])
            self.history['scores'].append(previous_score)
            self.history['n_selected'].append(len(self.selected_features))

        all_score = np.mean(cross_val_score(
            self.estimator, self.data, self.label.values.ravel(), scoring=self.metric, cv=5
        ))
        self.my_print("The cv score using all features is %.4lf" % all_score)
        self.my_print("The cv score using selected features is %.4lf" % previous_score)
        self.my_print("The number of selected features is [%d/%d] = %d%%" % (self.history['n_selected'][-1],
                                                                             total,
                                                                             int(self.history['n_selected'][-1] / total * 100)))

    def transform(self, data, label=None):
        return data[self.selected_features]

    def fit_transform(self, data, label, **fit_params):
        self.fit(data=data, label=label)
        return self.transform(data)

    def visualize(self, test_scores=None, all_score=None):
        if test_scores is not None:
            assert len(test_scores) == len(self.history['scores'])
        plt.figure(figsize=(12,6))
        plt.rcParams['font.size'] = 15
        plt.plot(self.history['n_examined'], self.history['scores'], label='cv scores on selected features: %.4lf' % self.history['scores'][-1])
        if test_scores is not None:
            plt.plot(self.history['n_examined'], test_scores, label='test scores on selected features: %.4lf' % test_scores[-1])
        if all_score is not None:
            plt.plot(self.history['n_examined'], [all_score] * len(self.history['scores']),
                     linestyle='--', label='test scores on all features: %.4lf' % all_score)
        plt.xlabel('The number of features examined')
        plt.ylabel(self.metric)
        plt.legend()

    def data_to_dataframe(self):
        try:
            if not isinstance(self.data, pd.DataFrame):
                self.data = pd.DataFrame(self.data)
            if not isinstance(self.label, pd.DataFrame):
                self.label = pd.DataFrame(self.label, index=self.data.index)
        except Exception as e:
            raise ValueError(f"Cannot transform data and label into dataframe due to error: {e}")

    def my_print(self, print_info):
        if self.verbose:
            print(print_info)

    def get_task(self):
        if self.task is None:
            if self.label[self.label.columns[0]].nunique() < 20:
                self.task = 'classification'
            else:
                self.task = 'regression'
            self.my_print(f"The task is detected as {self.task}.")
        else:
            assert self.task in ['classification', 'regression']

    def get_metric(self):
        if self.metric is None:
            if self.task == 'classification':
                if self.label[self.label.columns[0]].nunique() > 2:
                    self.metric = 'accuracy'
                else:
                    self.metric = 'roc_auc'
            else:
                self.metric = 'r2'
            self.my_print(f"The metric is automatically determined as {self.metric}.")
        else:
            assert self.metric in sklearn.metrics.SCORERS.keys()

    def get_estimator(self):
        if self.estimator is None:
            params = {'n_jobs': self.n_jobs, 'importance_type': 'gain', 'n_estimators': 200}
            if self.task == 'classification':
                self.estimator = lgb.LGBMClassifier(**params)
            else:
                self.estimator = lgb.LGBMRegressor(**params)


class TwoStageFeatureSelector:
    def __init__(self,
                 task: str = None,
                 train_index=None,
                 val_index=None,
                 categorical_features=None,
                 metric=None,
                 n_data_blocks=1,
                 min_features=0.2,
                 stage1_metric='predictive',
                 stage2_metric='gain_importance',
                 stage2_params=None,
                 n_repeats=1,
                 tmp_save_path='./openfe_tmp_data_xx.feather',
                 n_jobs=1,
                 seed=1,
                 verbose=True
                 ):
        ''' Feature Selection Using the two-stage pruning algorithm of OpenFE

        Parameters
        ----------
        task: str, optional (default=None)
            'classification' or 'regression', if None, label with n_unique_values less than 20
            will be set to classification, else regression.

        train_index: pd.index, optional (default=None)
            the index of the data for training purposes.

        val_index: pd.index, optional (default=None)
            the index of the data for validation purposes. If train_index or val_index is None,
            we split the data into 0.8 (train) and 0.2 (val). It is recommended to pass in the index
            if the data has time series property.

        candidate_features_list: list, optional (default=None)
            the candidate features list for filtering. If None, it will be generated
            automatically, and users can define their candidate features list according to
            their prior knowledge.

        init_scores: pd.DataFrame, optional (default=None)
            the initial scores for feature boosting. Please see our paper for more details. If None,
            we generate initial scores by 5-fold cross-validation.

        categorical_features: list, optional (default=None)
            a list of categorical features. If None, we detect categorical features by using
            data.select_dtypes(exclude=np.number).columns.to_list().

        metric: str, optional (default=None)
            The metric for evaluating the performance of new features in feature boosting. Currently
            support ['binary_logloss', 'multi_logloss', 'auc', 'rmse']. The default metric is
            'binary_logloss' for binary-classification, 'multi_logloss' for multi-classification,
            and 'rmse' for regression tasks.

        n_data_blocks: int, optional (default=8)
            The number of data blocks for successive feature-wise halving. See more details in our
            paper. Should be 2^k (e.g., 1, 2, 4, 8, 16, 32, ...). Larger values for faster speed,
            but may hurt the overall performance, especially when there are many useful
            candidate features.

        min_features: float, optional (default=0.2)
            The minimum number of features in percentage after successive feature-wise halving.
            It is used to early-stop successive feature-wise halving. When the number of
            features is smaller than min_features, successive
            feature-wise halving will stop immediately.

        stage1_metric: str, optional (default='predictive')
            The metric used for evaluating the features in stage1. Currently support
            ['predictive', 'corr', 'mi']. 'predictive' is the method described in the paper.
            'corr' is the Pearson correlation between the feature and the target.
            'mi' is the mutual information between the feature and the target.
            It is recommended to use the default 'predictive'.

        stage2_metric: str, optional (default='gain_importance')
            The feature importance used to rank the features in stage2. Currently support
            ['gain_importance', 'permutation'].
            'gain_importance' is the same as the importance in LightGBM.
            'permutation' is another feature importance method. It is sometimes better than
            gain importance, but requires much more computational time.

        stage2_params: dict, optional (default=None)
            The parameters for training LightGBM in stage2.

        n_repeats: int, optional (default=1)
            The number of repeats in permutation. Only useful when stage2_metric is set to 'permutation'.

        tmp_save_path: str, optional (default='./openfe_tmp_data.feather')
            Temporary path to save data for multiprocessing.

        n_jobs: int, optional (default=1)
            The number of processes used for feature calculation and evaluation.

        seed: int, optional (default=1)
            Random number seed. This will seed everything.

        verbose: bool, optional (default=True)
            Whether to display information.

        Returns
        -------
        new_features_list: list
            a list of new features, sorted by their importance (from most important to least important).
        '''
        assert stage2_metric in ['gain_importance', 'permutation']
        assert stage1_metric in ['predictive', 'corr', 'mi']
        if metric: assert metric in ['binary_logloss', 'multi_logloss', 'auc', 'rmse']
        np.random.seed(seed)
        random.seed(seed)

        self.task = task
        self.train_index = train_index
        self.val_index = val_index
        self.categorical_features = categorical_features
        self.metric = metric
        self.n_data_blocks = n_data_blocks
        assert 0 < min_features <= 1
        self.min_features = min_features
        self.stage1_metric = stage1_metric
        self.stage2_metric = stage2_metric
        self.stage2_params = stage2_params
        self.n_repeats = n_repeats
        self.tmp_save_path = tmp_save_path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose


    def fit(self, data, label):
        self.data = data
        self.label = label
        self.data_to_dataframe()
        self.task = self.get_task(self.task)
        self.process_label()
        self.process_and_save_data()

        self.metric = self.get_metric(self.metric)
        self.categorical_features = self.get_categorical_features(self.categorical_features)
        self.train_index, self.val_index = self.get_index(self.train_index, self.val_index)
        self.init_scores = self.get_init_score(None)

        self.candidate_features_list = self.data.columns.to_list()
        self.min_features = int(self.min_features * len(self.candidate_features_list))
        self.myprint(f"The number of candidate features is {len(self.candidate_features_list)}")
        self.myprint("Start stage I selection.")
        self.candidate_features_list = self.stage1_select()
        self.myprint(f"The number of remaining candidate features is {len(self.candidate_features_list)}")
        self.myprint("Start stage II selection.")
        self.new_features_scores_list = self.stage2_select()
        self.new_features_list = [feature for feature, _ in self.new_features_scores_list]
        os.remove(self.tmp_save_path)
        gc.collect()
        return self.new_features_list

    def myprint(self, s):
        if self.verbose:
            print(s)

    def process_label(self):
        if self.task == "regression":
            pass
        else:
            self.label[self.label.columns[0]] = self.label[self.label.columns[0]].astype('category').cat.codes

    def process_and_save_data(self):
        self.data.index.name = 'openfe_index'
        self.data.reset_index().to_feather(self.tmp_save_path)

    def get_index(self, train_index, val_index):
        if train_index is None or val_index is None:
            if self.task == 'classification':
                _, _, train_y, test_y = train_test_split(self.data, self.label, stratify=self.label,
                                                         test_size=0.2, random_state=self.seed)
            else:
                _, _, train_y, test_y = train_test_split(self.data, self.label, test_size=0.2, random_state=self.seed)
            return train_y.index, test_y.index
        else:
            return train_index, val_index


    def get_categorical_features(self, categorical_features):
        if categorical_features is None:
            return list(self.data.select_dtypes(exclude=np.number))
        else:
            return categorical_features

    def get_task(self, task):
        if task is None:
            if self.label[self.label.columns[0]].nunique() < 20:
                self.task = 'classification'
            else:
                self.task = 'regression'
            return self.task
        else:
            return task

    def get_metric(self, metric):
        if metric is None:
            if self.task == 'classification':
                if self.label[self.label.columns[0]].nunique() > 2:
                    return 'multi_logloss'
                else:
                    return 'binary_logloss'
            else:
                return 'rmse'
        else:
            return metric

    def data_to_dataframe(self):
        try:
            if not isinstance(self.data, pd.DataFrame) or not isinstance(self.label, pd.DataFrame):
                warnings.warn("data and label should both be pd.DataFrame and have the same index!!!")
            if not isinstance(self.data, pd.DataFrame):
                self.data = pd.DataFrame(self.data)
            if not isinstance(self.label, pd.DataFrame):
                self.label = pd.DataFrame(self.label, index=self.data.index)
        except Exception as e:
            raise ValueError(f"Cannot transform data and label into dataframe due to error: {e}")

    def get_init_score(self, init_scores, use_train=False):
        if init_scores is None:
            assert self.task in ["regression", "classification"]
            if self.task == 'regression':
                init_scores = np.array([np.mean(self.label.values.ravel())] * len(self.label))
            elif self.label[self.label.columns[0]].nunique() > 2:
                prob = self.label[self.label.columns[0]].value_counts().sort_index().to_list()
                prob = prob / np.sum(prob)
                prob = [list(prob)]
                init_scores = np.array(prob * len(self.label))
            else:
                def logit(x):
                    return np.log(x / (1 - x))

                init_scores = np.array([logit(np.mean(self.label.values.ravel()))] * len(self.label))
            init_scores = pd.DataFrame(init_scores, index=self.label.index)
        else:
            self.check_init_scores(init_scores)
        return init_scores

    def check_init_scores(self, init_scores):
        if self.task == 'classification':
            if ((init_scores[:100].values >= 0) & (init_scores[:100].values <= 1)).all():
                warnings.warn("The init_scores for classification should be raw scores instead of probability."
                              " But the init_scores are between 0 and 1.")

    def stage1_select(self, ratio=0.5):
        train_index_samples = self._subsample(self.train_index, self.n_data_blocks)
        val_index_samples = self._subsample(self.val_index, self.n_data_blocks)
        idx = 0
        train_idx = train_index_samples[idx]
        val_idx = val_index_samples[idx]
        idx += 1
        results = self._calculate_and_evaluate(self.candidate_features_list, train_idx, val_idx)
        candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)
        # candidate_features_scores = self.delete_same(candidate_features_scores)

        while idx != len(train_index_samples):
            n_reserved_features = max(int(len(candidate_features_scores) * ratio),
                                      min(len(candidate_features_scores), self.min_features))
            train_idx = train_index_samples[idx]
            val_idx = val_index_samples[idx]
            idx += 1
            if n_reserved_features <= self.min_features:
                train_idx = train_index_samples[-1]
                val_idx = val_index_samples[-1]
                idx = len(train_index_samples)
                self.myprint("Meet early-stopping in successive feature-wise halving.")
            candidate_features_list = [item[0] for item in candidate_features_scores[:n_reserved_features]]
            del candidate_features_scores[n_reserved_features:]
            gc.collect()

            results = self._calculate_and_evaluate(candidate_features_list, train_idx, val_idx)
            candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)

        return_results = [item[0] for item in candidate_features_scores if item[1] > 0]
        if not return_results:
            return_results = [item[0] for item in candidate_features_scores[:100]]
        return return_results

    def stage2_select(self):
        train_y = self.label.loc[self.train_index]
        val_y = self.label.loc[self.val_index]
        train_init = self.init_scores.loc[self.train_index]
        val_init = self.init_scores.loc[self.val_index]

        train_x = self.data[self.candidate_features_list].loc[self.train_index].copy()
        val_x = self.data[self.candidate_features_list].loc[self.val_index].copy()
        gc.collect()
        self.myprint("Finish data processing.")
        if self.stage2_params is None:
            params = {"n_estimators": 1000, "importance_type": "gain", "num_leaves": 16,
                      "seed": 1, "n_jobs": self.n_jobs}
        else:
            params = self.stage2_params
        if self.metric is not None:
            params.update({"metric": self.metric})
        if self.task == 'classification':
            gbm = lgb.LGBMClassifier(**params)
        else:
            gbm = lgb.LGBMRegressor(**params)
        gbm.fit(train_x, train_y.values.ravel(), init_score=train_init,
                eval_init_score=[val_init],
                eval_set=[(val_x, val_y.values.ravel())],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        results = []
        if self.stage2_metric == 'gain_importance':
            for i, imp in enumerate(gbm.feature_importances_):
                results.append([self.candidate_features_list[i], imp])
        elif self.stage2_metric == 'permutation':
            r = permutation_importance(gbm, val_x, val_y,
                                       n_repeats=self.n_repeats, random_state=self.seed, n_jobs=self.n_jobs)
            for i, imp in enumerate(r.importances_mean):
                results.append([self.candidate_features_list[i], imp])
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results

    def get_init_metric(self, pred, label):
        if self.metric == 'binary_logloss':
            init_metric = log_loss(label, scipy.special.expit(pred), labels=[0, 1])
        elif self.metric == 'multi_logloss':
            init_metric = log_loss(label, scipy.special.softmax(pred, axis=1),
                                   labels=list(range(pred.shape[1])))
        elif self.metric == 'rmse':
            init_metric = mean_squared_error(label, pred, squared=False)
        elif self.metric == 'auc':
            init_metric = roc_auc_score(label, scipy.special.expit(pred))
        else:
            raise NotImplementedError(f"Metric {self.metric} is not supported. "
                                      f"Please select metric from ['binary_logloss', 'multi_logloss'"
                                      f"'rmse', 'auc'].")
        return init_metric

    def delete_same(self, candidate_features_scores, threshold=1e-20):
        start_n = len(candidate_features_scores)
        if candidate_features_scores:
            pre_score = candidate_features_scores[0][1]
        else:
            return candidate_features_scores
        i = 1
        while i < len(candidate_features_scores):
            now_score = candidate_features_scores[i][1]
            if abs(now_score - pre_score) < threshold:
                candidate_features_scores.pop(i)
            else:
                pre_score = now_score
                i += 1
        end_n = len(candidate_features_scores)
        self.myprint(f"{start_n - end_n} same features have been deleted.")
        return candidate_features_scores

    def _subsample(self, iterators, n_data_blocks):
        iterators = list(iterators)
        length = int(len(iterators) / n_data_blocks)
        random.shuffle(iterators)
        results = [iterators[:length]]
        while n_data_blocks != 1:
            n_data_blocks = int(n_data_blocks / 2)
            length = int(length * 2)
            if n_data_blocks == 1:
                results.append(iterators)
            else:
                results.append(iterators[:length])
        return results

    def _evaluate(self, data_temp, candidate_feature, train_y, val_y, train_init, val_init, init_metric):
        try:
            train_x = pd.DataFrame(data_temp[candidate_feature].loc[train_y.index])
            val_x = pd.DataFrame(data_temp[candidate_feature].loc[val_y.index])
            if self.stage1_metric == 'predictive':
                params = {"n_estimators": 100, "importance_type": "gain", "num_leaves": 16,
                          "seed": 1, "deterministic": True, "n_jobs": 1}
                if self.metric is not None:
                    params.update({"metric": self.metric})
                if self.task == 'classification':
                    gbm = lgb.LGBMClassifier(**params)
                else:
                    gbm = lgb.LGBMRegressor(**params)
                gbm.fit(train_x, train_y.values.ravel(), init_score=train_init,
                        eval_init_score=[val_init],
                        eval_set=[(val_x, val_y.values.ravel())],
                        callbacks=[lgb.early_stopping(3, verbose=False)])
                key = list(gbm.best_score_['valid_0'].keys())[0]
                if self.metric in ['auc']:
                    score = gbm.best_score_['valid_0'][key] - init_metric
                else:
                    score = init_metric - gbm.best_score_['valid_0'][key]
            elif self.stage1_metric == 'corr':
                score = np.corrcoef(pd.concat([train_x, val_x], axis=0).fillna(0).values.ravel(),
                                    pd.concat([train_y, val_y], axis=0).fillna(0).values.ravel())[0, 1]
                score = abs(score)
            elif self.stage1_metric == 'mi':
                if self.task == 'regression':
                    r = mutual_info_regression(
                        pd.concat([train_x, val_x], axis=0).replace([np.inf, -np.inf], 0).fillna(0),
                        pd.concat([train_y, val_y], axis=0).values.ravel())
                else:
                    r = mutual_info_classif(pd.concat([train_x, val_x], axis=0).replace([np.inf, -np.inf], 0).fillna(0),
                                            pd.concat([train_y, val_y], axis=0).values.ravel())
                score = r[0]
            else:
                raise NotImplementedError("Cannot recognize filter_metric %s." % self.stage1_metric)
            return score
        except:
            print(traceback.format_exc())
            exit()

    def _calculate_multiprocess(self, candidate_features, train_idx, val_idx):
        try:
            results = []
            base_features = {'openfe_index'}
            for candidate_feature in candidate_features:
                base_features |= set(candidate_feature.get_fnode())

            data = pd.read_feather(self.tmp_save_path, columns=list(base_features)).set_index('openfe_index')
            data_temp = data.loc[train_idx + val_idx]
            del data
            gc.collect()

            for candidate_feature in candidate_features:
                candidate_feature.calculate(data_temp, is_root=True)
                candidate_feature.f_delete()
                results.append(candidate_feature)
            return results
        except:
            print(traceback.format_exc())
            exit()

    def _calculate(self, candidate_features, train_idx, val_idx):
        results = []
        length = int(np.ceil(len(candidate_features) / self.n_jobs / 4))
        n = int(np.ceil(len(candidate_features) / length))
        random.shuffle(candidate_features)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            with tqdm(total=n) as progress:
                for i in range(n):
                    if i == (n - 1):
                        future = ex.submit(self._calculate_multiprocess,
                                           candidate_features[i * length:],
                                           train_idx, val_idx)
                    else:
                        future = ex.submit(self._calculate_multiprocess,
                                           candidate_features[i * length:(i + 1) * length],
                                           train_idx, val_idx)
                    future.add_done_callback(lambda p: progress.update())
                    results.append(future)
                res = []
                for r in results:
                    res.extend(r.result())
        return res

    def _calculate_and_evaluate_multiprocess(self, candidate_features, train_idx, val_idx):
        try:
            results = []
            base_features = {'openfe_index'}
            for candidate_feature in candidate_features:
                base_features.add(candidate_feature)

            data = pd.read_feather(self.tmp_save_path, columns=list(base_features)).set_index('openfe_index')
            data_temp = data.loc[train_idx + val_idx]
            del data
            gc.collect()

            train_y = self.label.loc[train_idx]
            val_y = self.label.loc[val_idx]
            train_init = self.init_scores.loc[train_idx]
            val_init = self.init_scores.loc[val_idx]
            init_metric = self.get_init_metric(val_init, val_y)
            for candidate_feature in candidate_features:
                score = self._evaluate(data_temp, candidate_feature, train_y, val_y, train_init, val_init, init_metric)
                results.append([candidate_feature, score])
            return results
        except:
            print(traceback.format_exc())
            exit()

    def _calculate_and_evaluate(self, candidate_features, train_idx, val_idx):
        results = []
        length = int(np.ceil(len(candidate_features) / self.n_jobs / 4))
        n = int(np.ceil(len(candidate_features) / length))
        random.shuffle(candidate_features)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            with tqdm(total=n) as progress:
                for i in range(n):
                    if i == (n - 1):
                        future = ex.submit(self._calculate_and_evaluate_multiprocess,
                                           candidate_features[i * length:],
                                           train_idx, val_idx)
                    else:
                        future = ex.submit(self._calculate_and_evaluate_multiprocess,
                                           candidate_features[i * length:(i + 1) * length],
                                           train_idx, val_idx)
                    future.add_done_callback(lambda p: progress.update())
                    results.append(future)
                res = []
                for r in results:
                    res.extend(r.result())
        return res





