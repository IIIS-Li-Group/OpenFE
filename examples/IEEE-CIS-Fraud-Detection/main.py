import xgboost as xgb
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from openfe import openfe, get_candidate_features
from utils import transform, node_to_formula, formula_to_node
import warnings
from IEEE_utils import *
import random
import scipy.special
from multiprocessing import cpu_count
warnings.filterwarnings("ignore")
random.seed(1)
np.random.seed(1)
warnings.filterwarnings("ignore")

def prepare_data():
    # The codes for data preparation is from https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600
    # COLUMNS WITH STRINGS
    str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
                'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30',
                'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
    str_type += ['id-12', 'id-15', 'id-16', 'id-23', 'id-27', 'id-28', 'id-29', 'id-30',
                'id-31', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38']

    # FIRST 53 COLUMNS
    cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
           'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
           'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
           'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
           'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
           'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
           'M5', 'M6', 'M7', 'M8', 'M9']

    # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
    # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
    v =  [1, 3, 4, 6, 8, 11]
    v += [13, 14, 17, 20, 23, 26, 27, 30]
    v += [36, 37, 40, 41, 44, 47, 48]
    v += [54, 56, 59, 62, 65, 67, 68, 70]
    v += [76, 78, 80, 82, 86, 88, 89, 91]

    # v += [96, 98, 99, 104] # relates to groups, no NAN
    v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
    v += [124, 127, 129, 130, 136] # relates to groups, no NAN

    # LOTS OF NAN BELOW
    v += [138, 139, 142, 147, 156, 162] #b1
    v += [165, 160, 166] #b1
    v += [178, 176, 173, 182] #b2
    v += [187, 203, 205, 207, 215] #b2
    v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
    v += [218, 223, 224, 226, 228, 229, 235] #b3
    v += [240, 258, 257, 253, 252, 260, 261] #b3
    v += [264, 266, 267, 274, 277] #b3
    v += [220, 221, 234, 238, 250, 271] #b3

    v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
    v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
    v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN
    #v += [332, 325, 335, 338] # b4 lots NAN

    cols += ['V'+str(x) for x in v]
    dtypes = {}
    for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]+\
        ['id-0'+str(x) for x in range(1,10)]+['id-'+str(x) for x in range(10,34)]:
            dtypes[c] = 'float32'
    for c in str_type: dtypes[c] = 'category'

    # LOAD TRAIN
    X_train = pd.read_csv('./data/train_transaction.csv',index_col='TransactionID', dtype=dtypes, usecols=cols+['isFraud'])
    train_id = pd.read_csv('./data/train_identity.csv',index_col='TransactionID', dtype=dtypes)
    X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)
    # LOAD TEST
    X_test = pd.read_csv('./data/test_transaction.csv',index_col='TransactionID', dtype=dtypes, usecols=cols)
    test_id = pd.read_csv('./data/test_identity.csv',index_col='TransactionID', dtype=dtypes)
    fix = {o:n for o, n in zip(test_id.columns, train_id.columns)}
    test_id.rename(columns=fix, inplace=True)
    X_test = X_test.merge(test_id, how='left', left_index=True, right_index=True)
    # TARGET
    y_train = X_train['isFraud'].copy()
    del train_id, test_id, X_train['isFraud']; x = gc.collect()
    # PRINT STATUS
    print('Train shape',X_train.shape,'test shape',X_test.shape)

    for i in range(1,16):
        if i in [1,2,3,5,9]: continue
        X_train['D'+str(i)] =  X_train['D'+str(i)] - X_train.TransactionDT/np.float32(24*60*60)
        X_test['D'+str(i)] = X_test['D'+str(i)] - X_test.TransactionDT/np.float32(24*60*60)


    for i,f in enumerate(X_train.columns):
        # FACTORIZE CATEGORICAL VARIABLES
        if (str(X_train[f].dtype)=='category')|(X_train[f].dtype=='object'):
            df_comb = pd.concat([X_train[f],X_test[f]],axis=0)
            df_comb,_ = df_comb.factorize(sort=True)
            if df_comb.max()>32000: print(f,'needs int32')
            X_train[f] = df_comb[:len(X_train)].astype('int16')
            X_test[f] = df_comb[len(X_train):].astype('int16')


    # COMBINE COLUMNS CARD1+ADDR1
    encode_CB('card1', 'addr1', df1=X_train, df2=X_test)

    import datetime
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    X_train['DT_M'] = X_train['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
    X_train['DT_M'] = (X_train['DT_M'].dt.year - 2017) * 12 + X_train['DT_M'].dt.month

    X_test['DT_M'] = X_test['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
    X_test['DT_M'] = (X_test['DT_M'].dt.year - 2017) * 12 + X_test['DT_M'].dt.month

    X_train['day'] = X_train.TransactionDT / (24 * 60 * 60)
    X_train['uid'] = X_train.card1_addr1.astype(str) + '_' + np.floor(X_train.day - X_train.D1).astype(str)

    X_test['day'] = X_test.TransactionDT / (24 * 60 * 60)
    X_test['uid'] = X_test.card1_addr1.astype(str) + '_' + np.floor(X_test.day - X_test.D1).astype(str)
    # encode_LE('uid', train=X_train, test=X_test)
    return X_train, X_test, y_train


def expert_FE(X_train, X_test):
    # ExpertFE is from https://www.kaggle.com/code/cdeotte/xgb-fraud-with-magic-0-9600
    # TRANSACTION AMT CENTS
    X_train['cents'] = (X_train['TransactionAmt'] - np.floor(X_train['TransactionAmt'])).astype('float32')
    X_test['cents'] = (X_test['TransactionAmt'] - np.floor(X_test['TransactionAmt'])).astype('float32')
    print('cents, ', end='')
    # FREQUENCY ENCODE: ADDR1, CARD1, CARD2, CARD3, P_EMAILDOMAIN
    encode_FE(X_train, X_test, ['addr1', 'card1', 'card2', 'card3', 'P_emaildomain'])
    encode_CB('card1_addr1', 'P_emaildomain', X_train, X_test)
    # FREQUENCY ENCODE
    encode_FE(X_train, X_test, ['card1_addr1', 'card1_addr1_P_emaildomain'])
    # GROUP AGGREGATE
    encode_AG(['TransactionAmt', 'D9', 'D11'], ['card1', 'card1_addr1', 'card1_addr1_P_emaildomain'], ['mean', 'std'],
              train_df=X_train, test_df=X_test, usena=True)

    # FREQUENCY ENCODE UID
    encode_FE(X_train, X_test, ['uid'])
    # AGGREGATE
    encode_AG(['TransactionAmt', 'D4', 'D9', 'D10', 'D15'], ['uid'], ['mean', 'std'],
              train_df=X_train, test_df=X_test, fillna=True, usena=True)
    # AGGREGATE
    encode_AG(['C' + str(x) for x in range(1, 15) if x != 3], ['uid'], ['mean'],
              train_df=X_train, test_df=X_test, fillna=True, usena=True)
    # AGGREGATE
    encode_AG(['M' + str(x) for x in range(1, 10)], ['uid'], ['mean'],
              train_df=X_train, test_df=X_test, fillna=True, usena=True)
    # AGGREGATE
    encode_AG2(['P_emaildomain', 'dist1', 'DT_M', 'id_02', 'cents'], ['uid'],
               train_df=X_train, test_df=X_test)
    # AGGREGATE
    encode_AG(['C14'], ['uid'], ['std'],
              train_df=X_train, test_df=X_test, fillna=True, usena=True)
    # AGGREGATE
    encode_AG2(['C13', 'V314'], ['uid'], train_df=X_train, test_df=X_test)
    # AGGREATE
    encode_AG2(['V127', 'V136', 'V309', 'V307', 'V320'], ['uid'], train_df=X_train, test_df=X_test)
    # NEW FEATURE
    X_train['outsider15'] = (np.abs(X_train.D1 - X_train.D15) > 3).astype('int8')
    X_test['outsider15'] = (np.abs(X_test.D1 - X_test.D15) > 3).astype('int8')
    return X_train, X_test


def automatic_FE(X_train, X_test, y_train):
    # Step 1: get prediction for feature boosting
    idxT = X_train.index[:3 * len(X_train) // 4]
    idxV = X_train.index[3 * len(X_train) // 4:]
    if load_init_scores:
        oof_proba = pd.read_csv('./results/oof_xgb.csv')
        oof_proba = oof_proba.set_index('TransactionID')
    else:
        oof_proba = train_and_predict(X_train.copy(), X_test.copy(), y_train.copy(), init=True)
    print('XGB96 OOF VAL=', roc_auc_score(y_train.loc[idxV], scipy.special.expit(oof_proba.loc[idxV])),
          log_loss(y_train.loc[idxV], scipy.special.expit(oof_proba.loc[idxV])))
    oof_proba.columns = ['isFraud']
    oof_proba = oof_proba['isFraud']
    oof_proba = scipy.special.expit(oof_proba)

    positive = y_train[y_train == 1]
    negative = y_train[y_train == 0].sample(n=3*len(positive))
    label = pd.DataFrame(pd.concat([positive, negative], axis=0))
    # cols.append('uid')

    # Step 2: automatically generate features
    categorical_features = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
                            'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29',
                            'id_30',
                            'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo',
                            'uid', 'card1_addr1']
    train_index = label.index[label.index.isin(idxT)]
    val_index = label.index[label.index.isin(idxV)]
    print("len train index", len(train_index), "len val index", len(val_index))

    to_remove = ['C3', 'M5', 'id_08', 'id_33', 'card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34',
                 'TransactionDT', 'D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14', 'DT_M', 'day', 'uid']
    to_remove.extend(['id_' + str(x) for x in range(22, 28)])
    params = {"n_estimators": 1000, "importance_type": "gain", "num_leaves": 64,
              "seed": 1, "n_jobs": n_jobs}

    ordinal_features = []
    for f in X_train.columns:
        if f not in categorical_features: ordinal_features.append(f)

    candidate_features_list = get_candidate_features(numerical_features=[],
                                                     categorical_features=categorical_features,
                                                     ordinal_features=ordinal_features)
    np.save('./all_candidate_features.npy', np.array([node_to_formula(node) for node in candidate_features_list]))
    ofe = openfe()
    features = ofe.fit(data=X_train.loc[label.index], label=label,
                       init_scores=oof_proba,
                       candidate_features_list=candidate_features_list,
                       metric='rmse',
                       train_index=train_index, val_index=val_index,
                       categorical_features=categorical_features,
                       min_candidate_features=30000,
                       stage2_params=params,
                       drop_columns=to_remove,
                       n_jobs=n_jobs, n_data_blocks=8, task='regression')

    new_features_list = [feature for feature, score in features[:600] if score > 0]
    print("We are using %d more features." % len(new_features_list))
    for feature in new_features_list:
        feature.delete()
    X_train, X_test = transform(X_train, X_test, new_features_list, n_jobs=n_jobs)
    return X_train, X_test


def train_and_predict(X_train, X_test, y_train, init=False, seed=1):
    for i, f in enumerate(X_train.columns):
        if (str(X_train[f].dtype) == 'category') | (X_train[f].dtype == 'object'):
            df_comb = pd.concat([X_train[f], X_test[f]], axis=0)
            df_comb, _ = df_comb.factorize(sort=True)
            if df_comb.max() > 32000: print(f, 'needs int32')
            X_train[f] = df_comb[:len(X_train)].astype('int16')
            X_test[f] = df_comb[len(X_train):].astype('int16')
        # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
        elif f not in ['TransactionAmt', 'TransactionDT']:
            mn = np.min((X_train[f].min(), X_test[f].min()))
            X_train[f] -= np.float32(mn)
            X_test[f] -= np.float32(mn)
            X_train[f].fillna(-1, inplace=True)
            X_test[f].fillna(-1, inplace=True)

    cols = list(X_train.columns)
    # FAILED TIME CONSISTENCY TEST
    for c in ['C3', 'M5', 'id_08', 'id_33']:
        cols.remove(c)
    for c in ['card4', 'id_07', 'id_14', 'id_21', 'id_30', 'id_32', 'id_34']:
        cols.remove(c)
    for c in ['id_' + str(x) for x in range(22, 28)]:
        cols.remove(c)

    cols.remove('TransactionDT')
    for c in ['D6', 'D7', 'D8', 'D9', 'D12', 'D13', 'D14']:
        cols.remove(c)
    for c in ['DT_M', 'day', 'uid']:
        cols.remove(c)

    print('NOW USING THE FOLLOWING',len(cols),'FEATURES.')

    # for k in range(0, 1001, 50):
        # CHRIS - TRAIN 75% PREDICT 25%
    idxT = X_train.index[:3*len(X_train)//4]
    idxV = X_train.index[3*len(X_train)//4:]

    clf = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        missing=-1,
        eval_metric='auc',
        nthread=n_jobs,
        tree_method='hist',
        random_state=seed
        # tree_method='gpu_hist'
    )
    h = clf.fit(X_train.loc[idxT, cols], y_train[idxT],
                eval_set=[(X_train.loc[idxV, cols], y_train[idxV])],
                verbose=50, early_stopping_rounds=100)

    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, cols)), columns=['Value', 'Feature'])
    feature_imp.to_csv('./feature_imp_autoFE.csv')

    del clf, h
    x = gc.collect()

    oof = np.zeros(len(X_train))
    preds = np.zeros(len(X_test))

    skf = GroupKFold(n_splits=6)
    for i, (idxT, idxV) in enumerate(skf.split(X_train, y_train, groups=X_train['DT_M'])):
        month = X_train.iloc[idxV]['DT_M'].iloc[0]
        print('Fold', i, 'withholding month', month)
        print(' rows of train =', len(idxT), 'rows of holdout =', len(idxV))
        clf = xgb.XGBClassifier(
            n_estimators=5000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.4,
            missing=-1,
            eval_metric='auc',
            # USE CPU
            nthread=n_jobs,
            tree_method='hist',
            random_state=seed
            # USE GPU
            # tree_method='gpu_hist'
        )
        h = clf.fit(X_train[cols].iloc[idxT], y_train.iloc[idxT],
                    eval_set=[(X_train[cols].iloc[idxV], y_train.iloc[idxV])],
                    verbose=100, early_stopping_rounds=200)
        clf._Booster.save_model('./results/model_%d.json' % i)
        oof[idxV] += clf.predict(X_train[cols].iloc[idxV], output_margin=True)
        preds += clf.predict_proba(X_test[cols])[:, 1] / skf.n_splits
        del h, clf
        x = gc.collect()
    print('#' * 20)
    print('XGB96 OOF CV=', roc_auc_score(y_train, oof), log_loss(y_train, scipy.special.expit(oof)))
    if init:
        X_train['oof'] = oof
        X_train[['oof']].to_csv('./results/oof_xgb.csv')
        return X_train[['oof']]

    sample_submission = pd.read_csv('./data/sample_submission.csv')
    sample_submission.isFraud = preds
    return sample_submission


if __name__ == '__main__':
    # Please first download data from https://www.kaggle.com/competitions/ieee-fraud-detection/data
    # and place the Kaggle data into './data/' folder.
    os.makedirs('./data/', exist_ok=True)
    os.makedirs('./results/', exist_ok=True)
    n_jobs = min(40, cpu_count())

    load_init_scores = True
    use_openfe = True # use OpenFE to generate new features.
    use_expertfe = False # use the features generated by the first-place team.
    if use_openfe:
        X_train, X_test, y_train = prepare_data()
        print("Finish preparing data.")

        X_train, X_test = automatic_FE(X_train, X_test, y_train)
        features = X_train.columns[:217+600]
        sample_submission = train_and_predict(X_train[features], X_test[features], y_train)

        _save = './results/sub_xgb_OpenFE.csv'
        sample_submission.to_csv(_save, index=False)
    if use_expertfe:
        X_train, X_test, y_train = prepare_data()
        X_train, X_test = expert_FE(X_train, X_test)
        sample_submission = train_and_predict(X_train, X_test, y_train)
        _save = './results/sub_xgb_expertFE.csv'
        sample_submission.to_csv(_save, index=False)
