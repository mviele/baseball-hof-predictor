#Python 3

import pandas as pd
import numpy as np
import math
from xgboost import XGBClassifier as xg
from matplotlib import pyplot
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
    feature_import = [0.0 for d in range(0, (X_train.shape[1]))]

    for n in range (0, len(seed)):
        model.set_params(seed = seed[n])
        model.fit(X_train, y_train)
        for i in range(0, (X_train.shape[1])):
            feature_import[i] += model.feature_importances_[i]
        preds = model.predict_proba(X_cv)[:,1]
        for j in range (0, (X_cv.shape[0])):
            predictions[j] += preds[j]
    
    for i in range (0, len(feature_import)):
        feature_import[i] /= float(len(seed))
    for j in range (0, len(predictions)):
        predictions[j] /= float(len(seed))

    return np.array(predictions), np.array(feature_import)


def scale_games(X_test, average_games):
    
    for p in X_test.index:
        scale_val = average_games / X_test.loc[p, 'G']
        for f in list(X_test):
            if f in ['ERA', 'Mitchell-Report', 'Positive-Test', 'MVP', 'CyYoung', 'WorldSeriesMVP', 'GoldGlove']:
                continue
            X_test.loc[p, f] *= scale_val
    
def main():

    model = xg(learning_rate=0.095, gamma=0.8, max_depth=5, subsample=0.9,
                min_child_weight=0.8, colsample_bytree=0.5,
                objective='binary:logistic', seed=seed)

    print ('Reading data...')
    X_train = pd.read_csv('combined_stats_train.csv', sep=',', header=0)
    df = X_train.loc[X_train['HOF'] == 1]
    avg_games = math.floor(df['G'].mean())
    print("Average number of games played by HOFer: "+str(avg_games))
    X_train = X_train.set_index('playerID')
    y_train = X_train.ix[:,'HOF']
    X_train.drop('HOF', axis=1, inplace=True)

    X_test = pd.read_csv('combined_stats_test.csv', sep=',', header=0)
    X_test = X_test.set_index('playerID')
    id_test = X_test.index.values
    

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

        preds, feats = estimator_bagging(model, X_train_part, y_train_part, x_crossval)

        roc_auc = roc_auc_score(y_crossval, preds)
        mean_auc += roc_auc

        i += 1

    mean_auc /= num_folds
    print ('AUC: ' + str(mean_auc))


    print ('Training...')
    print ('Bagging parameters:')
    print ('    Number of trees: %d' % (num_trees))
    preds, feats = estimator_bagging(model, X_train, y_train, X_test)
    make_submission('predictions', id_test, preds)

    # plot
    pyplot.bar(range(len(feats)), feats)
    pyplot.xticks(range(len(feats)), list(X_test), rotation=90)
    pyplot.title("Feature Importance")
    pyplot.show()

    scale_games(X_test, avg_games)
    X_test.to_csv("scaled_stats.csv")

    preds, feats = estimator_bagging(model, X_train, y_train, X_test)
    make_submission('scaled-predictions', id_test, preds)

    print ('')
    print ('Program complete!')
    print ('')

main()

