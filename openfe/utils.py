import traceback
from .FeatureGenerator import Node, FNode
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np


def tree_to_formula(tree):
    if isinstance(tree, Node):
        if tree.name in ['+', '-', '*', '/']:
            string_1 = tree_to_formula(tree.children[0])
            string_2 = tree_to_formula(tree.children[1])
            return str('(' + string_1 + tree.name + string_2 + ')')
        else:
            result = [tree.name, '(']
            for i in range(len(tree.children)):
                string_i = tree_to_formula(tree.children[i])
                result.append(string_i)
                result.append(',')
            result.pop()
            result.append(')')
            return ''.join(result)
    elif isinstance(tree, FNode):
        return str(tree.name)
    else:
        return str(tree.name)


def formula_to_tree(string):
    if string[-1] != ')':
        return FNode(string)

    def is_trivial_char(c):
        return not (c in '()+-*/,')

    def find_prev(string):
        if string[-1] != ')':
            return max([(0 if is_trivial_char(c) else i+1) for i,c in enumerate(string)])
        level, pos = 0, -1
        for i in range(len(string)-1,-1,-1):
            if string[i] == ')': level += 1
            if string[i] == '(': level -= 1
            if level == 0:
                pos = i
                break
        while (pos>0) and is_trivial_char(string[pos-1]):
            pos -= 1
        return pos
    p2 = find_prev(string[:-1])
    if string[p2-1] == '(':
        return Node(string[:p2-1], [formula_to_tree(string[p2:-1])])
    p1 = find_prev(string[:p2-1])
    if string[0] == '(':
        return Node(string[p2-1], [formula_to_tree(string[p1:p2 - 1]), formula_to_tree(string[p2:-1])])
    else:
        return Node(string[:p1-1], [formula_to_tree(string[p1:p2 - 1]), formula_to_tree(string[p2:-1])])


def file_to_node(path):
    text = open(path,'r').read().split('\n')
    res = []
    for s in text:
        a = s.split(' ')
        if len(a)<=1: continue
        if len(a[0])==0 or a[0][-1]!=')': continue
        res.append([formula_to_tree(a[0]), float(a[1])])
    return res


def check_xor(node1, node2):
    def _get_FNode(node):
        if isinstance(node, FNode):
            return [node.name]
        else:
            res = []
            for child in node.children:
                res.extend(_get_FNode(child))
            return res
    fnode1 = set(_get_FNode(node1))
    fnode2 = set(_get_FNode(node2))
    if len(fnode1 ^ fnode2) == 0:
        return False
    else:
        return True


def split_num_cat_features(features_list):
    num_features = []
    cat_features = []
    for node in features_list:
        if node.name == 'Combine':
            cat_features.append(node)
        else:
            num_features.append(node)
    return num_features, cat_features


def _cal(feature, n_train):
    try:
        base_features = ['openfe_index']
        base_features.extend(feature.get_fnode())
        _data = pd.read_feather('./openfe_tmp_data.feather', columns=base_features).set_index('openfe_index')
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


def transform(X_train, X_test, new_features_list, n_jobs, name=""):
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
    data.reset_index().to_feather('./openfe_tmp_data.feather')
    n_train = len(X_train)
    ex = ProcessPoolExecutor(n_jobs)
    results = []
    for feature in new_features_list:
        results.append(ex.submit(_cal, feature, n_train))
    ex.shutdown(wait=True)
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
    return _train, _test


def rename_columns(df):
    return df.rename(columns={col: ('autoFE-%d' % i) for i, col in enumerate(df.columns)})
