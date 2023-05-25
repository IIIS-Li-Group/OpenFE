import gc
import os
import warnings
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from .FeatureGenerator import *
import random
from concurrent.futures import ProcessPoolExecutor
import traceback
from .utils import tree_to_formula, check_xor, formula_to_tree
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
import scipy.special
from copy import deepcopy
from tqdm import tqdm
# import tracemalloc
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)


def _enumerate(current_order_num_features, lower_order_num_features,
               current_order_cat_features, lower_order_cat_features):
    num_candidate_features = []
    cat_candidate_features = []
    for op in all_operators:
        for f in current_order_num_features+current_order_cat_features:
            num_candidate_features.append(Node(op, children=[deepcopy(f)]))
    for op in num_operators:
        for f in current_order_num_features:
            num_candidate_features.append(Node(op, children=[deepcopy(f)]))
    for op in num_num_operators:
        for i in range(len(current_order_num_features)):
            f1 = current_order_num_features[i]
            k = i if op in symmetry_operators else 0
            for f2 in current_order_num_features[k:] + lower_order_num_features:
                if check_xor(f1, f2):
                    num_candidate_features.append(Node(op, children=[deepcopy(f1), deepcopy(f2)]))

    for op in cat_num_operators:
        for f in current_order_num_features:
            for cat_f in current_order_cat_features + lower_order_cat_features:
                if check_xor(f, cat_f):
                    num_candidate_features.append(Node(op, children=[deepcopy(f), deepcopy(cat_f)]))
        for f in lower_order_num_features:
            for cat_f in current_order_cat_features:
                if check_xor(f, cat_f):
                    num_candidate_features.append(Node(op, children=[deepcopy(f), deepcopy(cat_f)]))

    for op in cat_cat_operators:
        for i in range(len(current_order_cat_features)):
            f1 = current_order_cat_features[i]
            k = i if op in symmetry_operators else 0
            for f2 in current_order_cat_features[k:] + lower_order_cat_features:
                if check_xor(f1, f2):
                    if op in ['Combine']:
                        cat_candidate_features.append(Node(op, children=[deepcopy(f1), deepcopy(f2)]))
                    else:
                        num_candidate_features.append(Node(op, children=[deepcopy(f1), deepcopy(f2)]))
    return num_candidate_features, cat_candidate_features


def get_candidate_features(numerical_features=None, categorical_features=None, ordinal_features=None, order=1):
    ''' You can determine the list of candidate features yourself. This function returns a list of
    candidate features that can be fed into openfe.fit(candidate_features_list)

    Parameters
    ----------
    numerical_features: list, optional (default=None)
        The numerical features in the data.

    categorical_features: list, optional (default=None)
        The categorical features in the data.

    ordinal_features: list, optional (default=None)
        Ordinal features are numerical features with discrete values, such as age.
        Ordinal features are treated as both numerical and categorical features when generating
        candidate features.

    order: int, optional (default=1)
        The maximum order of the generated candidate features. A value larger than 1 may result
        in an extremely large number of candidate features.

    Returns
    -------
    candidate_features_list: list
        A list of candidate features.
    '''
    if numerical_features is None: numerical_features = []
    if categorical_features is None: categorical_features = []
    if ordinal_features is None: ordinal_features = []
    assert len(set(numerical_features) & set(categorical_features) & set(ordinal_features)) == 0
    num_features = []
    cat_features = []
    for f in numerical_features+categorical_features+ordinal_features:
        if f in ordinal_features:
            num_features.append(FNode(f))
            cat_features.append(FNode(f))
        elif f in categorical_features:
            cat_features.append(FNode(f))
        else:
            num_features.append(FNode(f))

    current_order_num_features = num_features
    current_order_cat_features = cat_features
    lower_order_num_features = []
    lower_order_cat_features = []
    candidate_features_list = []

    while order > 0:
        _num, _cat = _enumerate(current_order_num_features, lower_order_num_features,
                                         current_order_cat_features, lower_order_cat_features)
        candidate_features_list.extend(_num)
        candidate_features_list.extend(_cat)
        lower_order_num_features, lower_order_cat_features = current_order_num_features, current_order_cat_features
        current_order_num_features, current_order_cat_features = _num, _cat
        order -= 1
    return candidate_features_list


def _subsample(iterators, n_data_blocks):
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


class OpenFE:
    def __init__(self):
        pass

    def fit(self,
            data: pd.DataFrame, label: pd.DataFrame,
            task: str = None,
            train_index=None,
            val_index=None,
            candidate_features_list=None,
            init_scores=None,
            categorical_features=None,
            metric=None, drop_columns=None,
            n_data_blocks=8,
            min_candidate_features=2000,
            feature_boosting=False,
            stage1_metric='predictive',
            stage2_metric='gain_importance',
            stage2_params=None,
            is_stage1=True,
            n_repeats=1,
            tmp_save_path='./openfe_tmp_data_xx.feather',
            n_jobs=1,
            seed=1,
            verbose=True):
        ''' Generate new features by the algorithm of OpenFE

        Parameters
        ----------
        data: pd.DataFrame
            the input data

        label: pd.DataFrame
            the target

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

        drop_columns: list, optional (default=None)
            A list of columns you would like to drop when building the LightGBM in stage2.
            These columns will still be used to generate candidate_features_list.

        n_data_blocks: int, optional (default=8)
            The number of data blocks for successive feature-wise halving. See more details in our
            paper. Should be 2^k (e.g., 1, 2, 4, 8, 16, 32, ...). Larger values for faster speed,
            but may hurt the overall performance, especially when there are many useful
            candidate features.

        min_candidate_features: int, optional (default=2000)
            The minimum number of candidate features after successive feature-wise halving.
            It is used to early-stop successive feature-wise halving. When the number of
            candidate features is smaller than min_candidate_features, successive
            feature-wise halving will stop immediately.

        feature_boosting: bool, optional (default=False)
            Whether to use feature boosting. See more details in our paper.
            If False, the init_scores will be set the same as the default values in LightGBM.

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

        is_stage1: bool, optional (default=True)
            Whether to use successive feature-wise halving to eliminate candidate features. If False,
            all the candidate features are calculated and used to train the LightGBM in stage2,
            which may require a large amount of memory as well as computational time.

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

        self.data = data
        self.label = label
        self.metric = metric
        self.drop_columns = drop_columns
        self.n_data_blocks = n_data_blocks
        self.min_candidate_features = min_candidate_features
        self.stage1_metric = stage1_metric
        self.stage2_metric = stage2_metric
        self.feature_boosting = feature_boosting
        self.stage2_params = stage2_params
        self.is_stage1 = is_stage1
        self.n_repeats = n_repeats
        self.tmp_save_path = tmp_save_path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.data_to_dataframe()
        self.task = self.get_task(task)
        self.process_label()
        self.process_and_save_data()

        self.metric = self.get_metric(metric)
        self.categorical_features = self.get_categorical_features(categorical_features)
        self.candidate_features_list = self.get_candidate_features(candidate_features_list)
        self.train_index, self.val_index = self.get_index(train_index, val_index)
        self.init_scores = self.get_init_score(init_scores)

        self.myprint(f"The number of candidate features is {len(self.candidate_features_list)}")
        self.myprint("Start stage I selection.")
        self.candidate_features_list = self.stage1_select()
        self.myprint(f"The number of remaining candidate features is {len(self.candidate_features_list)}")
        self.myprint("Start stage II selection.")
        self.new_features_scores_list = self.stage2_select()
        self.new_features_list = [feature for feature, _ in self.new_features_scores_list]
        for node, score in self.new_features_scores_list:
            node.delete()
        os.remove(self.tmp_save_path)
        gc.collect()
        return self.new_features_list

    def myprint(self, s):
        if self.verbose:
            print(s)

    # def monitor_memory(self):
    #     peak_memory = tracemalloc.get_traced_memory()
    #     print(peak_memory)
    #     self.myprint(f"Peak memory usage: {peak_memory[1] / 1024 / 1024:.2f} MB")

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

    def get_candidate_features(self, candidate_features_list):
        if candidate_features_list is None:
            ordinal_features = []
            numerical_features = []
            for feature in self.data.columns:
                if feature in self.categorical_features:
                    continue
                elif self.data[feature].nunique() <= 100:
                    ordinal_features.append(feature)
                else:
                    numerical_features.append(feature)
            candidate_features_list = get_candidate_features(
                numerical_features=numerical_features,
                categorical_features=self.categorical_features,
                ordinal_features=ordinal_features
            )
        return candidate_features_list

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
            if self.feature_boosting:
                data = self.data.copy()
                label = self.label.copy()

                params = {"n_estimators": 10000, "learning_rate": 0.1, "metric": self.metric,
                          "seed": self.seed, "n_jobs": self.n_jobs}
                if self.task == "regression":
                    gbm = lgb.LGBMRegressor(**params)
                else:
                    gbm = lgb.LGBMClassifier(**params)

                for feature in self.categorical_features:
                    data[feature] = data[feature].astype('category')
                    data[feature] = data[feature].cat.codes
                    data[feature] = data[feature].astype('category')

                if self.task == 'classification' and label[label.columns[0]].nunique() > 2:
                    init_scores = np.zeros((len(data), label[label.columns[0]].nunique()))
                else:
                    init_scores = np.zeros(len(data))
                skf = StratifiedKFold(n_splits=5) if self.task == "classification" else KFold(n_splits=5)
                for train_index, val_index in skf.split(data, label):
                    X_train, y_train = data.iloc[train_index], label.iloc[train_index]
                    X_val, y_val = data.iloc[val_index], label.iloc[val_index]

                    gbm.fit(X_train, y_train.values.ravel(),
                            eval_set=[[X_val, y_val.values.ravel()]], callbacks=[lgb.early_stopping(200)])

                    if use_train:
                        init_scores[train_index] += (gbm.predict_proba(X_train, raw_score=True) if self.task == "classification" else \
                                                 gbm.predict(X_train)) / (skf.n_splits - 1)
                    else:
                        init_scores[val_index] = gbm.predict_proba(X_val, raw_score=True) if self.task == "classification" else \
                            gbm.predict(X_val)

                init_scores = pd.DataFrame(init_scores, index=data.index)
            else:
                if self.task == 'regression':
                    init_scores = np.array([np.mean(self.label.values.ravel())]*len(self.label))
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
            if ((init_scores[:100].values>=0)&(init_scores[:100].values<=1)).all():
                warnings.warn("The init_scores for classification should be raw scores instead of probability."
                              " But the init_scores are between 0 and 1.")

    def stage1_select(self, ratio=0.5):
        if self.is_stage1 is False:
            train_index = _subsample(self.train_index, self.n_data_blocks)[0]
            val_index = _subsample(self.val_index, self.n_data_blocks)[0]
            self.data = self.data.loc[train_index+val_index]
            self.label = self.label.loc[train_index+val_index]
            self.train_index = train_index
            self.val_index = val_index
            return [[f, 0] for f in self._calculate(self.candidate_features_list, train_index, val_index)]
        train_index_samples = _subsample(self.train_index, self.n_data_blocks)
        val_index_samples = _subsample(self.val_index, self.n_data_blocks)
        idx = 0
        train_idx = train_index_samples[idx]
        val_idx = val_index_samples[idx]
        idx += 1
        results = self._calculate_and_evaluate(self.candidate_features_list, train_idx, val_idx)
        candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)
        candidate_features_scores = self.delete_same(candidate_features_scores)

        while idx != len(train_index_samples):
            n_reserved_features = max(int(len(candidate_features_scores)*ratio),
                                      min(len(candidate_features_scores), self.min_candidate_features))
            train_idx = train_index_samples[idx]
            val_idx = val_index_samples[idx]
            idx += 1
            if n_reserved_features <= self.min_candidate_features:
                train_idx = train_index_samples[-1]
                val_idx = val_index_samples[-1]
                idx = len(train_index_samples)
                self.myprint("Meet early-stopping in successive feature-wise halving.")
            candidate_features_list = [item[0] for item in candidate_features_scores[:n_reserved_features]]
            del candidate_features_scores[n_reserved_features:]; gc.collect()

            results = self._calculate_and_evaluate(candidate_features_list, train_idx, val_idx)
            candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)

        return_results = [item[0] for item in candidate_features_scores if item[1] > 0]
        if not return_results:
            return_results = [item[0] for item in candidate_features_scores[:100]]
        return return_results

    def stage2_select(self):
        data_new = []
        new_features = []
        self.candidate_features_list = self._calculate(self.candidate_features_list,
                                                       self.train_index.to_list(),
                                                       self.val_index.to_list())
        index_tmp = self.candidate_features_list[0].data.index
        for feature in self.candidate_features_list:
            new_features.append(tree_to_formula(feature))
            data_new.append(feature.data.values)
            feature.delete()
        gc.collect()
        data_new = np.vstack(data_new)
        data_new = pd.DataFrame(data_new.T, index=index_tmp,
                                columns=['autoFE-%d' % i for i in range(len(new_features))])
        data_new = pd.concat([data_new, self.data], axis=1)
        for f in self.categorical_features:
            data_new[f] = data_new[f].astype('category')
            data_new[f] = data_new[f].cat.codes
            data_new[f] = data_new[f].astype('category')
        data_new = data_new.replace([np.inf, -np.inf], np.nan)
        if self.drop_columns is not None:
            data_new = data_new.drop(self.drop_columns, axis=1)
        train_y = self.label.loc[self.train_index]
        val_y = self.label.loc[self.val_index]
        train_init = self.init_scores.loc[self.train_index]
        val_init = self.init_scores.loc[self.val_index]

        train_x = data_new.loc[self.train_index].copy()
        val_x = data_new.loc[self.val_index].copy()
        del data_new
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
            for i, imp in enumerate(gbm.feature_importances_[:len(new_features)]):
                results.append([formula_to_tree(new_features[i]), imp])
        elif self.stage2_metric == 'permutation':
            r = permutation_importance(gbm, val_x, val_y,
                                       n_repeats=self.n_repeats, random_state=self.seed, n_jobs=self.n_jobs)
            for i, imp in enumerate(r.importances_mean[:len(new_features)]):
                results.append([formula_to_tree(new_features[i]), imp])
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
        self.myprint(f"{start_n-end_n} same features have been deleted.")
        return candidate_features_scores

    def _evaluate(self, candidate_feature, train_y, val_y, train_init, val_init, init_metric):
        try:
            train_x = pd.DataFrame(candidate_feature.data.loc[train_y.index])
            val_x = pd.DataFrame(candidate_feature.data.loc[val_y.index])
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
                    r = mutual_info_regression(pd.concat([train_x, val_x], axis=0).replace([np.inf, -np.inf], 0).fillna(0),
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
        # for f in candidate_features:
        #     f.delete()
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
                base_features |= set(candidate_feature.get_fnode())

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
                candidate_feature.calculate(data_temp, is_root=True)
                score = self._evaluate(candidate_feature, train_y, val_y, train_init, val_init, init_metric)
                candidate_feature.delete()
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
        for f in candidate_features:
            f.delete()
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            with tqdm(total=n) as progress:
                for i in range(n):
                    if i == (n-1):
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

    def _trans(self, feature, n_train):
        try:
            base_features = ['openfe_index']
            base_features.extend(feature.get_fnode())
            _data = pd.read_feather(self.tmp_save_path, columns=base_features).set_index('openfe_index')
            feature.calculate(_data, is_root=True)
            if (str(feature.data.dtype) == 'category') | (str(feature.data.dtype) == 'object'):
                pass
            else:
                feature.data = feature.data.replace([-np.inf, np.inf], np.nan)
                # feature.data = feature.data.fillna(0)
        except:
            print(traceback.format_exc())
            exit()
        return ((str(feature.data.dtype) == 'category') or (str(feature.data.dtype) == 'object')), \
               feature.data.values.ravel()[:n_train], \
               feature.data.values.ravel()[n_train:], \
               tree_to_formula(feature)

    def transform(self, X_train, X_test, new_features_list, n_jobs, name=""):
        """ Transform train and test data according to new features. Since there are global operators such as
        'GroupByThenMean', train and test data need to be transformed together.

        :param X_train: pd.DataFrame, the train data
        :param X_test:  pd.DataFrame, the test data
        :param new_features_list: the new features to transform data.
        :param n_jobs: the number of processes to calculate data
        :param name: used for naming new features
        :return: X_train, X_test. The transformed train and test data.
        """
        if len(new_features_list) == 0:
            return X_train, X_test

        data = pd.concat([X_train, X_test], axis=0)
        data.index.name = 'openfe_index'
        data.reset_index().to_feather(self.tmp_save_path)
        n_train = len(X_train)

        self.myprint("Start transforming data.")
        start = datetime.now()
        ex = ProcessPoolExecutor(n_jobs)
        results = []
        for feature in new_features_list:
            results.append(ex.submit(self._trans, feature, n_train))
        ex.shutdown(wait=True)
        self.myprint(f"Time spent calculating new features {datetime.now()-start}.")
        _train = []
        _test = []
        names = []
        names_map = {}
        cat_feats = []
        for i, res in enumerate(results):
            is_cat, d1, d2, f = res.result()
            names.append('autoFE_f_%d' % i + name)
            names_map['autoFE_f_%d' % i + name] = f
            _train.append(d1)
            _test.append(d2)
            if is_cat: cat_feats.append('autoFE_f_%d' % i + name)
        _train = np.vstack(_train)
        _test = np.vstack(_test)
        _train = pd.DataFrame(_train.T, columns=names, index=X_train.index)
        _test = pd.DataFrame(_test.T, columns=names, index=X_test.index)
        for c in _train.columns:
            if c in cat_feats:
                _train[c] = _train[c].astype('category')
                _test[c] = _test[c].astype('category')
            else:
                _train[c] = _train[c].astype('float')
                _test[c] = _test[c].astype('float')
        _train = pd.concat([X_train, _train], axis=1)
        _test = pd.concat([X_test, _test], axis=1)
        self.myprint("Finish transformation.")
        os.remove(self.tmp_save_path)
        return _train, _test



