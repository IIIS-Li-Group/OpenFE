from cmath import nan
import numpy as np
import pandas as pd

all_operators = ["freq"]
num_operators = ["abs", "log", "sqrt", "square", "sigmoid", "round", "residual"]
num_num_operators = ["min", "max", "+", "-", "*", "/"]
cat_num_operators = ["GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
                     "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank"]
cat_cat_operators = ["Combine", "CombineThenFreq", "GroupByThenNUnique"]

symmetry_operators = ["min", "max", "+", "-", "*", "/", "Combine", "CombineThenFreq"]
cal_all_operators = ["freq",
                     "GroupByThenMin", "GroupByThenMax", "GroupByThenMean",
                     "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank",
                     "Combine", "CombineThenFreq", "GroupByThenNUnique"]


class Node(object):
    def __init__(self, op, children):
        self.name = op
        self.children = children
        self.data = None
        self.train_idx = []
        self.val_idx = []

    def get_fnode(self):
        fnode_list = []
        for child in self.children:
            fnode_list.extend(child.get_fnode())
        return list(set(fnode_list))

    def delete(self):
        self.data = None
        for child in self.children:
            child.delete()

    def f_delete(self):
        for child in self.children:
            child.f_delete()

    def calculate(self, data, is_root=False):
        if self.name in all_operators+num_operators:
            d = self.children[0].calculate(data)
            if self.name == "abs":
                new_data = d.abs()
            elif self.name == "log":
                new_data = np.log(np.abs(d.replace(0, np.nan)))
            elif self.name == "sqrt":
                new_data = np.sqrt(np.abs(d))
            elif self.name == "square":
                new_data = np.square(d)
            elif self.name == "sigmoid":
                new_data = 1 / (1 + np.exp(-d))
            elif self.name == "freq":
                value_counts = d.value_counts()
                value_counts.loc[np.nan] = np.nan
                new_data = d.apply(lambda x: value_counts.loc[x])
            elif self.name == "round":
                new_data = np.floor(d)
            elif self.name == "residual":
                new_data = d - np.floor(d)
            else:
                raise NotImplementedError(f"Unrecognized operator {self.name}.")
        elif self.name in num_num_operators:
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            if self.name == "max":
                new_data = np.maximum(d1, d2)
            elif self.name == "min":
                new_data = np.minimum(d1, d2)
            elif self.name == "+":
                new_data = d1 + d2
            elif self.name == "-":
                new_data = d1 - d2
            elif self.name == "*":
                new_data = d1 * d2
            elif self.name == "/":
                new_data = d1 / d2.replace(0, np.nan)
        else:
            d1 = self.children[0].calculate(data)
            d2 = self.children[1].calculate(data)
            if self.name == "GroupByThenMin":
                temp = d1.groupby(d2).min()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMax":
                temp = d1.groupby(d2).max()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMean":
                temp = d1.groupby(d2).mean()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenMedian":
                temp = d1.groupby(d2).median()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == "GroupByThenStd":
                temp = d1.groupby(d2).std()
                temp.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: temp.loc[x])
            elif self.name == 'GroupByThenRank':
                new_data = d1.groupby(d2).rank(ascending=True, pct=True)
            elif self.name == "GroupByThenFreq":
                def _f(x):
                    value_counts = x.value_counts()
                    value_counts.loc[np.nan] = np.nan
                    return x.apply(lambda x: value_counts.loc[x])
                new_data = d1.groupby(d2).apply(_f)
            elif self.name == "GroupByThenNUnique":
                nunique = d1.groupby(d2).nunique()
                nunique.loc[np.nan] = np.nan
                new_data = d2.apply(lambda x: nunique.loc[x])
            elif self.name == "Combine":
                temp = d1.astype(str) + '_' + d2.astype(str)
                temp[d1.isna() | d2.isna()] = np.nan
                temp, _ = temp.factorize()
                new_data = pd.Series(temp, index=d1.index).astype("float64")
            elif self.name == "CombineThenFreq":
                temp = d1.astype(str) + '_' + d2.astype(str)
                temp[d1.isna() | d2.isna()] = np.nan
                value_counts = temp.value_counts()
                value_counts.loc[np.nan] = np.nan
                new_data = temp.apply(lambda x: value_counts.loc[x])
            else:
                raise NotImplementedError(f"Unrecognized operator {self.name}.")
        if self.name == 'Combine':
            new_data = new_data.astype('category')
        else:
            new_data = new_data.astype('float')
        if is_root:
            self.data = new_data
        return new_data



class FNode(object):
    def __init__(self, name):
        self.name = name
        self.data = None
        self.calculate_all = False

    def delete(self):
        self.data = None

    def f_delete(self):
        self.data = None

    def get_fnode(self):
        return [self.name]

    def calculate(self, data):
        self.data = data[self.name]
        return self.data
