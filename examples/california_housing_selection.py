import sys
sys.path.append('../')
import pandas as pd
from sklearn.datasets import fetch_california_housing
from openfe import OpenFE, tree_to_formula, transform, TwoStageFeatureSelector
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


def get_score(train_x, test_x, train_y, test_y):
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1)
    params = {'n_estimators': 1000, 'n_jobs': n_jobs, 'seed': 1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score


if __name__ == '__main__':
    n_jobs = 4
    data = fetch_california_housing(as_frame=True).frame
    label = data[['MedHouseVal']]
    del data['MedHouseVal']

    train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=1)
    # get baseline score
    score = get_score(train_x, test_x, train_y, test_y)
    print("The MSE before feature generation is", score)
    # We use the two-stage pruning algorithm of OpenFE to perform Feature Selection
    fs = TwoStageFeatureSelector(n_jobs=n_jobs)
    features = fs.fit(data=train_x, label=train_y)

    # OpenFE gives the ranking of the base features:
    print(features)
    # Select the top 6 features
    new_features = features[:6]
    score = get_score(train_x[new_features], test_x[new_features], train_y, test_y)
    print("The MSE after feature selection is", score)

