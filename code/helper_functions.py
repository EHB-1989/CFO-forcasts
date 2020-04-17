#!/usr/bin/env python
# coding: utf-8


from math import sqrt
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# build the features and target variable using the lenght of sequence history and lenght of target sequence

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, single_step=False):
    
  '''
    Create inputs array with 3D shapes to feed the LSTM model
  '''
  data = []
  labels = []

  start_index = start_index + history_size

  if end_index is None:
     end_index = len(dataset) - target_size+1
  #if end_index is None:
  else :
        end_index = len(dataset[:end_index]) - target_size
  
  

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


    
def split_train_test(df, date_train_val_split, past_history=3, future_target=12):
    '''
    Split the dataset into a train and validation sets
    input :datafram
    output : datafram
    
    date_train_val_split : the last row to use for training in the dataset
    
    '''
    
    '''
    Split the dataset into a train and validation sets
    input :datafram
    output : datafram
    
    date_train_val_split : the last row to use for training in the dataset
    
    '''
    
    last_date = date_train_val_split  # < to use for train

    # create df_train and df_val 
    df_train = df.loc[:last_date]
    lock_back_date = len(df.loc[ :last_date])-past_history
    df_test = df.iloc[lock_back_date:]

    #add rows with nan values if lenght is less than the (past_history+future_target), this is necessary 
    #to match to input of the model
    size_val_set = len(df_test)-(past_history+future_target)
    df_column_lenght = len(df.columns)
    if size_val_set<0 :
        nan_array = np.full((-size_val_set,df_column_lenght), np.nan)
        # get the last index of df_test and converet it to string
        index_min = df_test.index[-1:]+ pd.to_timedelta(31,unit='days') # move the next month
        index_min = df_test.index[-1:].strftime('%Y-%m-%d').to_series()[:1][0]
        timestamp = pd.date_range(start='2019-04-01' , periods=(-size_val_set), freq='MS')
        lenght = len(df.columns)
        df_test = pd.concat([df_test, (pd.DataFrame(nan_array,index=timestamp))],axis=1).iloc[:,:-lenght]
        
    return df_train, df_test



# a function to use to plot the numpy array (feature vs target)
def create_time_steps(length):
  return list(range(-length, 0))


# A function to plot each prediction vs real values


def multi_step_plot(history, true_future, prediction, target_index, generic_lockup_variables:List):
  '''
   plot the predictions for a variable for a given sequence from history
   
  '''
    
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, target_index]), label='History')
  plt.plot(np.arange(num_out), np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out), np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.title(generic_lockup_variables[target_index])
  plt.show()


def plot_train_history(history, title):
  loss = history.history['loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  
  plt.title(title)
  plt.legend()

  plt.show()
    
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Returns mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



def max_absolute_percentage_error(y_true, y_pred):
    """
    Returns max absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.max(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    """
    Returns root mean square error
    """
    return sqrt(mean_squared_error(y_true, y_pred))



def create_csv(forcast, columns_names,title:str, index_min ='2019-04-01', future_target=12 , dir_path=None):
    '''
        Create a dataframe from forcast and output csv file with date and generic lockup key in header
    '''
    df_forcast = pd.concat([pd.DataFrame(forcast[i] ) for i in range(len(forcast))], axis=1)
    df_forcast.columns = columns_names 
    #create date
    timestamp = pd.date_range(start=index_min, periods=future_target, freq='MS')
    df_forcast.index = timestamp
    df_forcast.index.name = 'Date'
    
    sub_file_name =title + '.csv'
    dir_ouput = dir_path
    path_to_csv = os.path.join(dir_path,sub_file_name,)
    df_forcast.to_csv(path_or_buf=path_to_csv, na_rep='NAN',columns=columns_names, index=True)
    
    return df_forcast



