import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import kendalltau

from sklearn.model_selection import GridSearchCV
from IPython.display import display, clear_output
from res.seq2featuresV1 import Transformer

def get_CV_MCC(xData, yData, Train_indx, Test_indx, models, ProtVec):
    CV = []
    for i, model in enumerate(models):
        transformer = Transformer()
        transformer.set_modelList(model, ProtVec=ProtVec)
        transformer.set_data(xData, yData)

        xTrain = transformer.xData[Train_indx,:]
        yTrain = transformer.yData[Train_indx]

        xTest  = transformer.xData[Test_indx,:]
        yTest  = transformer.yData[Test_indx]
        
        scores = []
        for _ in range(5):
            clf =  SVR()
            
            score = cross_val_score(clf, xTrain, yTrain, cv=5, scoring=make_scorer(mean_absolute_error))
            scores.append(np.mean(score))
            
        mean_score = np.mean(scores)
        median_score = np.median(scores)


        runDetails = {key:value for key, value in model.__dict__.items() if key!='location'}
        if ProtVec is not None:
            runDetails['protVec'] = ProtVec
        else:
            runDetails['protVec'] = 'w/o'

        runDetails['mean_cv_mean_absolute_error'] = mean_score
        runDetails['median_cv_mean_absolute_error'] = median_score
        runDetails['runID'] = i

        CV.append(runDetails)
        df = pd.DataFrame.from_dict(CV, orient = 'columns')
        clear_output(wait = True)
        display(df)

    clear_output()
    return df

def get_test_score(xData, yData, Train_indx, Test_indx, models, ProtVec):
    testScore = []
    for i, model in enumerate(models):
        transformer = Transformer()
        transformer.set_modelList(model, ProtVec=ProtVec)
        transformer.set_data(xData, yData)

        xTrain = transformer.xData[Train_indx,:]
        yTrain = transformer.yData[Train_indx]

        xTest  = transformer.xData[Test_indx,:]
        yTest  = transformer.yData[Test_indx]

        if isinstance(model, list):
            runDetails = {f'Model{i}': str(mod) for i, mod in enumerate(model)}
        else:
            runDetails = {key:value for key, value in model.__dict__.items() if key!='location'}

        if ProtVec is not None:
            runDetails['protVec'] = ProtVec
        else:
            runDetails['protVec'] = 'w/o'

        result = getTestScore(xTrain, yTrain, xTest, yTest)
        runDetails = {**runDetails, **result}

        testScore.append(runDetails)

        dt = pd.DataFrame.from_dict(testScore, orient = 'columns')
        clear_output(wait = True)
        display(dt)

    clear_output()
    return dt

def getTestScore(xTrain, yTrain, xTest, yTest):

    estimator = SVR()
    param_grid = {
    'kernel': ['rbf'],
        'C': [0.1, 1,10, 100, 1000],
        'epsilon': [0.0001, 0.001, 0.01, 0.1, 1, 5, 10],
        'gamma': ['scale',0.0001, 0.001, 0.005, 0.1, 1, 3, 5],
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False) # as smaller error is better
    import warnings
    warnings.filterwarnings('ignore', message='invalid value encountered in double_scalars')

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        verbose=3,
        scoring=scorer,
        n_jobs=11)

    grid.fit(xTrain, yTrain)
    yPred = grid.predict(xTest)

    result = {'Train':len(yTrain),
              'Test' :len(yTest),

              'Parameters': ' '.join([f'{key}:{val}' for key, val in grid.best_params_.items()]),

              'Train_mean_absolute_error':(-1)*grid.best_score_,
              'mean_absolute_error': mean_absolute_error(yTest, yPred),
              'r2_score': r2_score(yTest, yPred),
              'kendalltau':kendalltau(yTest, yPred)[0]
        }
    return result


