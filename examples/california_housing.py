# import sys
# sys.path.append('../')
import pandas as pd
from sklearn.datasets import fetch_california_housing
from openfe import openfe, transform, tree_to_formula
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
    # feature generation
    ofe = openfe()
    features = ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)

    train_x, test_x = ofe.transform(train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs)
    score = get_score(train_x, test_x, train_y, test_y)
    print("The MSE after feature generation is", score)
    print("The top 10 generated features are")
    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))
