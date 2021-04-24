from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit
from IPython.display import display, clear_output
from res.seq2featuresV1 import Transformer, GetModels, W2V_Model
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

kf = StratifiedShuffleSplit(n_splits=5)
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
            clf = SVC(gamma = 'scale')
            
            score = cross_val_score(clf, xTrain, yTrain, cv=kf, scoring=make_scorer(matthews_corrcoef))
            scores.append(np.mean(score))
            
        mean_score = np.mean(scores)
        median_score = np.median(scores)

        runDetails = {key:value for key, value in model.__dict__.items() if key!='location'}

        if ProtVec is not None:
            runDetails['protVec'] = ProtVec
        else:
            runDetails['protVec'] = 'w/o'
            
        

        runDetails['mean_CV_MCC'] = mean_score
        runDetails['median_CV_MCC'] = median_score
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

    estimator = SVC()
    param_grid = {   
        'kernel': ['rbf'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'class_weight': ['balanced'],
        'gamma': [0.001, 0.01, 0.1, 1 , 'scale'],
    }

    scorer = make_scorer(matthews_corrcoef)
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
    cMat = confusion_matrix(yTest, yPred, labels=(1,0)).reshape(-1, )

    result = {'Train':len(yTrain),
              'Test' :len(yTest),

              'Parameters': ' '.join([f'{key}:{val}' for key, val in grid.best_params_.items()]),

              'Train_MCC':grid.best_score_,
              'TP': cMat[0],
              'FN': cMat[1],
              'FP': cMat[2],
              'TN': cMat[3],
              'Accuracy': accuracy_score(yTest, yPred),
              'MCC': matthews_corrcoef(yTest, yPred),
        }
    return result