#Python 3

import pandas as pd
import numpy as np
from xgboost import XGBClassifier as xg
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.cross_validation import StratifiedKFold

#First 37 Ramanujan primes
seed = [2, 11, 17, 29, 41, 47, 59, 67, 71, 97, 101, 107, 127, 149, 151, 167, 179, 181, 227, 229, 
        233, 239, 241, 263, 269, 281, 307, 311, 347, 349, 367, 373, 401, 409, 419, 431, 433]


def make_submission(csv_name, idx, preds):
    submission = pd.DataFrame({ 'id': idx,
                                'HOF': preds })
    submission.to_csv(csv_name + ".csv", index=False, columns = ['id', 'HOF'])


def estimator_bagging(model, X_train, y_train, X_cv):

    predictions = [0.0  for d in range(0, (X_cv.shape[0]))]

    for n in range (0, len(seed)):
         model.set_params(seed = seed[n])
         model.fit(X_train, y_train)
         preds = model.predict_proba(X_cv)[:,1]
         for j in range (0, (X_cv.shape[0])):
                 predictions[j] += preds[j]

    for j in range (0, len(predictions)):
                 predictions[j] /= float(len(seed))

    return np.array(predictions)


def main():

    model = xg(learning_rate=0.095, gamma=0.8, max_depth=5, subsample=0.9,
                min_child_weight=0.8, colsample_bytree=0.5,
                objective='binary:logistic', seed=seed)

    print ('Reading data...')
    X_train = pd.read_csv('combined_stats.csv', sep=',', header=0)
    X_train = X_train.set_index('playerID')
    y_train = X_train.ix[:,'HOF']
    X_train.drop('HOF', axis=1, inplace=True)
    

    print ('')
    print ('Training data: ')
    print ('Shape: ' + str(X_train.shape))
    print (X_train.head())
    print ('')


    num_folds = 5 
    num_trees = len(seed) 

    mean_auc = 0.0
    i = 0
    folds = StratifiedKFold(y_train, n_folds=num_folds, shuffle=True, random_state=seed[6])

    for trainIndex, testIndex in folds:
        X_train_part, x_crossval = X_train.iloc[trainIndex], X_train.iloc[testIndex]
        y_train_part, y_crossval = np.array(y_train)[trainIndex], np.array(y_train)[testIndex]

        preds = estimator_bagging(model, X_train_part, y_train_part, x_crossval)

        roc_auc = roc_auc_score(y_crossval, preds)
        mean_auc += roc_auc

        i += 1

    mean_auc /= num_folds
    print ('AUC: ' + str(mean_auc))


    # print ('Training...')
    # print ('Bagging parameters:')
    # print ('    Number of trees: %d' % (num_trees))
    # preds = estimator_bagging(model, X_train, y_train, X_test)
    # make_submission('SubmissionBagging37', id_test, preds)

    print ('')
    print ('Program complete!')
    print ('')

main()

