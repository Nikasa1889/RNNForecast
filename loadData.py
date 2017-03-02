import os.path
import pandas as pd
import numpy as np

#DateTime is the timeStamp of the time-series


def loadData(inputDir="GEFCom2012", maxDataPoints=-1):
    '''
    The function loadData search for the two files:
        training_set.csv: data with NAs rows to mark the forecasting periods
        complete.csv: complete data to measure performance
        without the complete.csv file, the function return None
    input:
        inputDir: directory contains the input csv files
        maxDataPoints: maximum number of data points used for training data, make it fair for each experiment
    return:
        trainingDfs: list of training dataframe
        completeDfs: list of complete dataframe
    '''
    trainingFile = inputDir+"training_set.csv"
    completeFile = inputDir+"complete.csv"

    def readTimeSeries(filename):
        df = pd.read_csv(filename, nrows = 20) #Sample the csv file
        dtypes = {col:np.float64 for col in df if col !="DateTime"} #Force float64 dtype for all collumn except DateTime
        df = pd.read_csv(filename, dtype = dtypes, parse_dates=["DateTime"])
        return df
    
    def findForecastingPeriod(trainingDf):
        '''Search for missing periods to do forecasting'''
        idxNaCases = trainingDf.isnull().any(axis=1).values #Convert to numpy array
        startPoints = np.where(idxNaCases & np.logical_not(np.append(False, idxNaCases[:-1])) & np.append(idxNaCases[1:], True))[0]
        endPoints = np.where(idxNaCases & np.append(True, idxNaCases[:-1]) & np.logical_not(np.append(idxNaCases[1:], False)))[0]
        return startPoints, endPoints


    if not os.path.isfile(trainingFile):
        print("Err: Can't find training_set.csv file inside "+inputDir)
    else:
        trainingDf = readTimeSeries(trainingFile)
        if not os.path.isfile(completeFile):
            print("Info: No complete.csv file, can only do forecasting ")
            completeDf = None
            return([trainingDf], None)
        else:
            completeDf = readTimeSeries(completeFile)
            startPoints, endPoints = findForecastingPeriod(trainingDf)
            trainingDfs = []
            completeDfs = []
            for (startPoint, endPoint) in zip(startPoints, endPoints):
                startIdx = 0
                if (maxDataPoints > 0):
                    startIdx = max(startPoint-maxDataPoints, 0)
                trainingDfs.append(trainingDf[startIdx:startPoint-1])
                completeDfs.append(completeDf[startIdx:endPoint])
            return(trainingDfs, completeDfs)