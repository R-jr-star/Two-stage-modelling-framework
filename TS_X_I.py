# -*- coding: utf-8 -*-

pip install tensorflow-determinism

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri

numpy2ri.activate()

import logging
logging.getLogger('rpy2.rinterface_lib.callbacks').setLevel(logging.ERROR)

# ro.r['install.packages']('forecast')
utils = rpackages.importr('utils')
utils.install_packages('forecast')

forecast_pkg = importr('forecast')
stats = importr('stats')
generics = importr('generics')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, Embedding
from tensorflow.keras.models import Sequential
# from tensorflow.keras import optimizers
import time
import tensorflow as tf
import copy

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
# from pmdarima.arima import auto_arima

from tqdm import tqdm
import random
import os

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def split_windows(x_lst, num_series, input_len, output_len):
  num_window_per_series = np.zeros(num_series, dtype=int)
  # Calculate the number of windows per series
  for i in range(num_series):
      num_window_per_series[i] = x_lst[i].shape[0] - input_len
  # Calculate the total number of windows
  num_window_total = np.sum(num_window_per_series)

  # Initialize the data frame
  x_data_frame = np.zeros((num_window_total, input_len + output_len))

  # Populate the data_frame
  win = 0
  for k in range(num_series):
      for win_k in range(num_window_per_series[k]):
          x_data_frame[win, :] = x_lst[k].loc[
              win_k:(win_k + input_len + output_len-1), 'data'
          ] # 0:30 include the last value
          win += 1

  # Initialize mean and standard deviation vectors
  mean_vec = np.zeros(num_window_total)
  std_vec = np.zeros(num_window_total)
  # Calculate mean and standard deviation for each window
  for win in range(num_window_total):
      mean_vec[win] = np.mean(x_data_frame[win, :input_len]) # :30, exclude the last one
      std_vec[win] = np.std(x_data_frame[win, :input_len])
  # Normalize the data frame
  norm_x_data_frame = np.zeros_like(x_data_frame)
  for win in range(num_window_total):
      norm_x_data_frame[win, :] = (
          x_data_frame[win, :] - mean_vec[win]
      ) / std_vec[win]

  return([x_data_frame, norm_x_data_frame, num_window_per_series, mean_vec, std_vec])

def pred_and_results(x, model, x_data_frame, output_len, mean_vec_x, std_vec_x):
  x_predict = model.predict(x).reshape(-1)
  x_true = x_data_frame[:, -output_len]
  # Initialize an array for the original predictions
  orig_pred_x = np.zeros(len(x_predict))

  # Transform predictions back to the original scale
  for i in range(len(x_predict)):
      orig_pred_x[i] = x_predict[i] * std_vec_x[i] + mean_vec_x[i]

  return([orig_pred_x, x_true])

def split_to_series(a, num_per_series, num_series):
  a_lst = []
  # Loop through each series
  for i in range(num_series):
      a_lst.append(a[:num_per_series[i]]) # exclude the last one
      a = a[num_per_series[i]:]  # Remove used elements from a
  return(a_lst)

def metrics(num_series, true_demand_lst, pred_test_demand_lst):
  # Initialize arrays to store metrics
  rmse_vec = np.zeros(num_series)
  mae_vec = np.zeros(num_series)
  smape_vec = np.zeros(num_series)

  # Compute metrics for each series
  for i in range(num_series):
      true_values = true_demand_lst[i]
      predicted_values = pred_test_demand_lst[i]

      # RMSE
      rmse_vec[i] = np.sqrt(np.mean((predicted_values - true_values) ** 2))

      # MAE
      mae_vec[i] = np.mean(np.abs(predicted_values - true_values))

      # sMAPE
      smape_vec[i] = 2 * np.mean(
          np.abs(predicted_values - true_values) /
          (np.abs(predicted_values) + np.abs(true_values))
      )

  # Aggregate results
  mean_rmse = np.mean(rmse_vec)
  median_rmse = np.median(rmse_vec)

  mean_mae = np.mean(mae_vec)
  median_mae = np.median(mae_vec)

  mean_smape = np.mean(smape_vec)
  median_smape = np.median(smape_vec)

  # Print results
  print("Mean RMSE:", mean_rmse)
  print("Median RMSE:", median_rmse)

  print("Mean MAE:", mean_mae)
  print("Median MAE:", median_mae)

  print("Mean sMAPE:", mean_smape)
  print("Median sMAPE:", median_smape)

  # cum_rmse
  cum_rmse_lst = [
    np.mean([
        np.sqrt(
            np.mean((np.array(true_demand_lst[i][:t+1]) - np.array(pred_test_demand_lst[i][:t+1]))**2)
        )
        for t in range(len(true_demand_lst[i]))
    ])
    for i in range(len(true_demand_lst))
  ]

  cum_mae_lst = [
    np.mean([
        np.mean(
            np.abs(np.array(true_demand_lst[i][:t+1]) - np.array(pred_test_demand_lst[i][:t+1]))
            )
        for t in range(len(true_demand_lst[i]))
    ])
    for i in range(len(true_demand_lst))
  ]

  cum_smape_lst = [
    np.mean([
        2* np.mean(
            np.abs(np.array(true_demand_lst[i][:t+1]) - np.array(pred_test_demand_lst[i][:t+1])) /
            (np.abs(np.array(true_demand_lst[i][:t+1])) + np.abs(np.array(pred_test_demand_lst[i][:t+1])))
            )
        for t in range(len(true_demand_lst[i]))
    ])
    for i in range(len(true_demand_lst))
  ]

  mean_cum_rmse = np.mean(cum_rmse_lst)
  median_cum_rmse = np.median(cum_rmse_lst)
  mean_cum_mae = np.mean(cum_mae_lst)
  median_cum_mae = np.median(cum_mae_lst)
  mean_cum_smape = np.mean(cum_smape_lst)
  median_cum_smape = np.median(cum_smape_lst)

  # print("Mean Cum RMSE:", mean_cum_rmse)
  # print("Median Cum RMSE:", median_cum_rmse)

  return([mean_rmse, median_rmse, mean_mae, median_mae, mean_smape, median_smape,
          mean_cum_rmse, median_cum_rmse, mean_cum_mae, median_cum_mae, mean_cum_smape, median_cum_smape])

def build_mlp_model(input_len, output_len):
  model = Sequential()
  model.add(Dense(units=16, activation='tanh', input_dim= input_len))
  # model.add(Dense(4))
  model.add(Dense(output_len))
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  print(model.summary())
  return(model)

def build_lstm_model(input_len, output_len):
  model = Sequential()
  model.add(LSTM(units = 16, return_sequences = False, input_shape=(input_len, 1)))
  model.add(Dropout(0.5))
  model.add(Dense(4))
  model.add(Dense(output_len))
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  print(model.summary())
  return(model)

def run_main_m3(data, info, input_len, output_len = 1, seasonal_period = 12,
                build_model = build_mlp_model, batch_size = 64, epochs = 30, random_seed = 42):
  num_series = info.shape[0]
  print('num_series:', num_series)

  data_lst = []
  for i in range(num_series):
    data_lst.append(data[data['idx']==(i+1)])

  xtest_idx = np.array(info['test_idx'])
  xtest_lst, xtrain_lst = [], []
  for i in range(num_series):
    xtrain_lst.append(data_lst[i][data_lst[i]['time']<xtest_idx[i]].reset_index(drop=True))
    xtest_lst.append(data_lst[i][data_lst[i]['time']>=xtest_idx[i] - input_len].reset_index(drop=True))

  train_data_frame, norm_train_data_frame, num_window_per_series_train, mean_vec_train, std_vec_train = \
  split_windows(xtrain_lst, num_series, input_len, output_len)
  # Calculate the total number of windows
  num_window_total_train = np.sum(num_window_per_series_train)
  print('total number of windows:', num_window_total_train)

  # [batch, timesteps, feature]
  train_X = norm_train_data_frame[:,:-output_len].reshape((-1,input_len,1))
  train_Y = norm_train_data_frame[:,-output_len].reshape((-1,output_len,1))

  print('start training')
  tf.keras.backend.clear_session()
  # Call the above function with seed value
  set_global_determinism(seed=random_seed)
  model = build_model(input_len, output_len)
  history = model.fit(train_X, train_Y, batch_size = batch_size, epochs = epochs, verbose=1, shuffle=False)

  orig_pred_train, train_true = \
  pred_and_results(train_X, model, train_data_frame, output_len, mean_vec_train, std_vec_train)
  train_rmse_without_adj = np.sqrt(np.mean((orig_pred_train - train_true)**2))
  print('train_rmse_without_adj:', train_rmse_without_adj)

  # adj models
  diff = orig_pred_train - train_true
  diff_lst = split_to_series(diff, num_window_per_series_train, num_series)
  pred_lst = split_to_series(orig_pred_train, num_window_per_series_train, num_series)

  # Initialize p-value vector
  p_vec = np.zeros(num_series)
  # Perform the Ljung-Box test for each series
  for i in range(num_series):
      lb_test = acorr_ljungbox(diff_lst[i], lags=int(np.log(len(diff_lst[i]))), return_df=True)
      p_vec[i] = lb_test['lb_pvalue'].iloc[-1]  # Get the p-value from the last lag
  # Count and get indices of series with p-values < 0.05
  hetero_series_idx = np.where(p_vec < 0.05)[0]
  print(f"Number of heteroscedastic series: {len(hetero_series_idx)}")

  # Initialize list for adjusted models
  adj_models = [0] * num_series

  # Adjust predictions for heteroscedastic series
  for i in tqdm(hetero_series_idx):
      # Fit ARIMA model
      ts_data = ro.r['ts'](ro.FloatVector(diff_lst[i]), frequency=seasonal_period)
      arima_model = forecast_pkg.auto_arima(ts_data)
      adj_models[i] = arima_model
      # Update predictions by subtracting fitted values from the ARIMA model
      adj_values = np.array(ro.r['forecast'](arima_model).rx2('fitted'))
      pred_lst[i] = pred_lst[i] - adj_values

  # Combine predictions back into a single array
  adj_forecast = np.concatenate(pred_lst)
  # Compute RMSE
  train_rmse_with_adj = np.sqrt(np.mean((adj_forecast - train_true) ** 2))
  print("train_rmse_with_adj:", train_rmse_with_adj)

  print('start testing')
  test_data_frame, norm_test_data_frame, num_window_per_series_test, mean_vec_test, std_vec_test = \
  split_windows(xtest_lst, num_series, input_len, output_len)
  true_demand_lst = split_to_series(test_data_frame[:,-1].reshape(-1), num_window_per_series_test, num_series)
  test_predict = model.predict(norm_test_data_frame[:, :input_len].reshape((-1,input_len,1)))
  orig_test_predict = test_predict.reshape(-1) * std_vec_test + mean_vec_test
  pred_test_demand_lst = split_to_series(orig_test_predict, num_window_per_series_test, num_series)
  print('metrics before adjusting')
  # res_without_adj = metrics(num_series, true_demand_lst, pred_test_demand_lst)
  pred_test_demand_lst_without_adj = copy.deepcopy(pred_test_demand_lst)

  # Adjusted forecast
  for i in hetero_series_idx:
    adj = np.array(generics.forecast(adj_models[i], h = num_window_per_series_test[i]).rx2('mean'))
    pred_test_demand_lst[i] = pred_test_demand_lst[i] - adj
  print('metrics after adjusting')
  # res_with_adj = metrics(num_series, true_demand_lst, pred_test_demand_lst)
  pred_test_demand_lst_with_adj = pred_test_demand_lst

  # metrics_res = {'res_without_adj': res_without_adj,
  #              'res_with_adj': res_with_adj}
  # df = pd.DataFrame(metrics_res)
  return([num_series, true_demand_lst, pred_test_demand_lst_without_adj, pred_test_demand_lst_with_adj])

def metrics_m3(num_series, true_demand_lst, pred_test_demand_lst_without_adj, pred_test_demand_lst_with_adj):
  res_without_adj = metrics(num_series, true_demand_lst, pred_test_demand_lst_without_adj)
  res_with_adj = metrics(num_series, true_demand_lst, pred_test_demand_lst_with_adj)
  metrics_res = {'res_without_adj': res_without_adj,
               'res_with_adj': res_with_adj}
  df = pd.DataFrame(metrics_res)
  return(df)



m3_1_data = pd.read_csv('m3_1_data.csv')
m3_1_info = pd.read_csv('m3_1_info.csv')

num_series_m3_1, true_demand_lst_m3_1, pred_test_demand_lst_without_adj_m3_1, pred_test_demand_lst_with_adj_m3_1 = \
run_main_m3(m3_1_data, m3_1_info, input_len = 24, output_len = 1, seasonal_period = 12,
            build_model = build_mlp_model, batch_size = 32, epochs = 30, random_seed = 3407)

metrics_m3(num_series_m3_1, true_demand_lst_m3_1,
           pred_test_demand_lst_without_adj_m3_1, pred_test_demand_lst_with_adj_m3_1)