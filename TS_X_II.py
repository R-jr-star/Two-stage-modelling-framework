# -*- coding: utf-8 -*-

pip install tensorflow-determinism

pip install tsfeatures

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, LSTM, Dense, Activation, Dropout, Embedding, Concatenate, Add, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
# from tensorflow.keras import optimizers
import time
import tensorflow as tf

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
# from pmdarima.arima import auto_arima

from tqdm import tqdm
import random
import os
import copy

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from tsfeatures import tsfeatures
from tsfeatures import acf_features, pacf_features, entropy, lumpiness, stl_features, arch_stat, nonlinearity, unitroot_kpss, unitroot_pp, holt_parameters, hw_parameters

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
  # cum_error
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

  return([mean_cum_rmse, median_cum_rmse, mean_cum_mae, median_cum_mae, mean_cum_smape, median_cum_smape])

def build_mlp_model(input_len, output_len):
  inputs = Input(shape=(input_len,))
  x = Dense(16, activation="tanh")(inputs)
  outputs = Dense(output_len)(x)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  print(model.summary())
  return(model)

def build_lstm_model(input_len, output_len):
  inputs = Input(shape=(input_len,1))
  x = LSTM(units = 16, return_sequences = False)(inputs)
  x = Dropout(0.5)(x)
  x = Dense(4)(x)
  outputs = Dense(output_len)(x)
  model = Model(inputs=inputs, outputs=outputs)
  model.compile(loss = 'mean_squared_error', optimizer = 'adam')
  print(model.summary())
  return(model)

def build_cluster_model_V1(model, input_len, output_len):
  input_layer_model_A = model.input
  output_model_A = model.layers[-2].output # Exclude the last layer / output layer
  x = Dense(input_len, activation='tanh', name='new_dense_1')(output_model_A)
  x = Add()([x, Reshape((input_len,))(input_layer_model_A)])
  x = Dense(2, activation='tanh', name='new_dense_2')(x)
  output_clu = Dense(output_len, name='new_dense_3')(x)
  model_clu = Model(inputs=input_layer_model_A, outputs=output_clu)

  for i in range(len(model.layers) - 1):  # Exclude the output layer of the global model
      model_clu.layers[i].set_weights(model.layers[i].get_weights())
  # Freeze layers
  for layer in model_clu.layers[:(len(model.layers) - 1)]:
      layer.trainable = False

  # model_clu.summary()
  return(model_clu)

def run_cluster_models_m3(required_num_of_clusters, hetero_series_idx, cluster_labels,
                       xtrain_lst, xtest_lst, num_series,
                       input_len, output_len,
                       model, epochs_stage_2, batch_size_stage_2,
                       pred_lst, train_true_lst,
                       test_pred_lst, test_true_lst,
                       num_window_per_series_train, num_window_per_series_test,
                       random_seed):
  count_cluster, num_windosw_cluster, num_hetero_cluster = [], [], []
  test_pred_with_adj_lst = copy.deepcopy(test_pred_lst)

  for idx_cluster in range(required_num_of_clusters):
    cluster_i = hetero_series_idx[np.where(cluster_labels == idx_cluster)]
    count_cluster.append(len(cluster_i))

    cluster_lst = []
    for i in cluster_i:
      cluster_lst.append(xtrain_lst[i])
    cluster_test_lst = []
    for i in cluster_i:
      cluster_test_lst.append(xtest_lst[i])

    # prepare training data
    train_data_frame_clu, norm_train_data_frame_clu, num_window_per_series_train_clu, mean_vec_train_clu, std_vec_train_clu = \
    split_windows(cluster_lst, len(cluster_i), input_len, output_len)
    num_windosw_cluster.append(np.sum(num_window_per_series_train_clu))
    # [batch, timesteps, feature]
    train_X_clu = norm_train_data_frame_clu[:,:-output_len].reshape((-1,input_len,1))
    train_Y_clu = norm_train_data_frame_clu[:,-output_len].reshape((-1,output_len,1))

    # if np.sum(num_window_per_series_train_clu) <= 4000:
    #   model_clu = build_cluster_model(model, output_len, complex=False)
    # else:
    #   model_clu = build_cluster_model(model, output_len, complex=True)
    model_clu = build_cluster_model_V1(model, input_len, output_len)
    print(model_clu.summary())
    # Compile the model
    model_clu.compile(loss = 'mean_squared_error', optimizer = 'adam')
    # Fine-tune the model
    set_global_determinism(seed=random_seed)
    model_clu.fit(train_X_clu, train_Y_clu, epochs=epochs_stage_2, batch_size=batch_size_stage_2)

    ##### evaluation on training dataset for this cluster
    train_pred_clu, train_true_clu = pred_and_results(train_X_clu, model_clu, train_data_frame_clu, output_len,
                                                      mean_vec_train_clu, std_vec_train_clu)
    train_pred_lst_clu = [pred_lst[i] for i in cluster_i]
    train_true_lst_clu = [train_true_lst[i] for i in cluster_i]
    # train, without adj
    train_metrics_clu_without_adj = metrics(len(cluster_i), train_true_lst_clu, train_pred_lst_clu)

    train_pred_clu_lst = split_to_series(train_pred_clu, num_window_per_series_train[cluster_i], len(cluster_i))
    # train, with adj
    train_metrics_clu_with_adj = metrics(len(cluster_i), train_true_lst_clu, train_pred_clu_lst)

    ##### detect heteroscedastic series for this cluster
    diff_lst_with_adj = [np.array([train_true_lst_clu[i] - train_pred_clu_lst[i]]).reshape(-1) for i in range(len(train_true_lst_clu))]
    # Initialize p-value vector
    p_vec_clu = np.zeros(len(cluster_i))
    # Perform the Ljung-Box test for each series
    for i in range(len(cluster_i)):
        lb_test = acorr_ljungbox(diff_lst_with_adj[i], lags=int(np.log(len(diff_lst_with_adj[i]))), return_df=True)
        p_vec_clu[i] = lb_test['lb_pvalue'].iloc[-1]  # Get the p-value from the last lag
    # Count and get indices of series with p-values < 0.05
    hetero_series_idx_clu = np.where(p_vec_clu < 0.05)[0]
    print(f"Number of heteroscedastic series: {len(hetero_series_idx_clu)}")
    num_hetero_cluster.append(len(hetero_series_idx_clu))

    ##### evaluation on test dataset for this cluster
    test_pred_lst_clu = [test_pred_lst[i] for i in cluster_i]
    test_true_lst_clu = [test_true_lst[i] for i in cluster_i]
    # test, without adj
    test_metrics_clu_without_adj = metrics(len(cluster_i), test_true_lst_clu, test_pred_lst_clu)

    # test, with adj
    test_data_frame_clu, norm_test_data_frame_clu, num_window_per_series_test_clu, mean_vec_test_clu, std_vec_test_clu = \
    split_windows(cluster_test_lst, len(cluster_i), input_len, output_len)
    test_X_clu = norm_test_data_frame_clu[:,:-output_len].reshape((-1,input_len,1))
    test_pred_clu, _ = pred_and_results(test_X_clu, model_clu, test_data_frame_clu, output_len,
                                        mean_vec_test_clu, std_vec_test_clu)
    test_pred_clu_lst = split_to_series(test_pred_clu, num_window_per_series_test[cluster_i], len(cluster_i))
    test_metrics_clu_with_adj = metrics(len(cluster_i), test_true_lst_clu, test_pred_clu_lst)

    ##### substitute and update the predictions of this cluster
    for i in cluster_i:
      test_pred_with_adj_lst[i] = test_pred_clu_lst[np.where(cluster_i == i)[0][0]]

    df = pd.DataFrame({'train_metrics_clu_without_adj': train_metrics_clu_without_adj,
                      'train_metrics_clu_with_adj': train_metrics_clu_with_adj,
                      'test_metrics_clu_without_adj': test_metrics_clu_without_adj,
                      'test_metrics_clu_with_adj': test_metrics_clu_with_adj})
    print('cluster', idx_cluster, df)

  # test_metrics_with_adj = metrics(num_series, test_true_lst, test_pred_with_adj_lst)
  print(count_cluster, num_windosw_cluster, num_hetero_cluster)
  return([num_series, test_true_lst, test_pred_with_adj_lst])

def stage_2_m3(xtrain_lst, xtest_lst, num_series, input_len, output_len,
            model, epochs_stage_2, batch_size_stage_2,
            pred_lst, train_true_lst, test_pred_lst, test_true_lst,
            hetero_series_idx, diff_lst,
            seasonal_period, random_seed,
            num_window_per_series_train, num_window_per_series_test,
            required_num_of_clusters = 5):
  set_global_determinism(seed=random_seed)
  print('clustering into', required_num_of_clusters, 'clusters')
  hetero_df = pd.DataFrame({'unique_id': [], 'ds': [], 'y': []})
  for i in hetero_series_idx:
    hetero_df_i = {'unique_id': [int(i) for j in range(len(diff_lst[i]))],
                  'ds': list(xtrain_lst[i].loc[(input_len):, 'time']),
                  'y': list(diff_lst[i])}
    hetero_df = pd.concat([hetero_df, pd.DataFrame(hetero_df_i)], axis=0, ignore_index=True)
  selected_features = tsfeatures(hetero_df, freq=seasonal_period,
                                 features=[acf_features, pacf_features, entropy, lumpiness, stl_features, arch_stat,
                                           nonlinearity, unitroot_kpss, unitroot_pp, holt_parameters, hw_parameters])
  # Select specific features
  selected_features = selected_features[[
      'hw_alpha', 'hw_beta', 'hw_gamma', 'alpha', 'beta',
      'trend', 'spike', 'linearity', 'curvature', 'entropy',
      'x_acf1', 'lumpiness'
  ]]
  # Apply Min-Max normalization
  scaler = MinMaxScaler()
  normalized_features = scaler.fit_transform(selected_features)
  normalized_features = pd.DataFrame(normalized_features, columns=selected_features.columns)
  # Remove columns containing NaN values
  normalized_features = normalized_features.dropna(axis=1)
  # divide domains
  kmeans = KMeans(n_clusters=required_num_of_clusters, random_state=0).fit(normalized_features)
  cluster_labels = kmeans.labels_

  print('building models for each cluster')
  num_series, test_true_lst, test_pred_with_adj_lst = run_cluster_models_m3(required_num_of_clusters, hetero_series_idx, cluster_labels,
                                             xtrain_lst, xtest_lst, num_series, input_len, output_len,
                                             model, epochs_stage_2, batch_size_stage_2,
                                             pred_lst, train_true_lst,
                                             test_pred_lst, test_true_lst,
                                             num_window_per_series_train, num_window_per_series_test,
                                             random_seed)
  return([num_series, test_true_lst, test_pred_with_adj_lst])

def run_global_domain_global_m3(data, info, input_len, output_len = 1, seasonal_period = 12,
                             load_model_name = None, build_model = build_lstm_model,
                             batch_size_stage_1 = 32, batch_size_stage_2 = 16,
                             epochs_stage_1 = 30, epochs_stage_2 = 40,
                             required_num_of_clusters = 5,
                             random_seed = 42):
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

  tf.keras.backend.clear_session()
  # Call the above function with seed value
  set_global_determinism(seed=random_seed)

  if load_model_name == None:
    print('start training')
    model = build_model(input_len, output_len)
    history = model.fit(train_X, train_Y, batch_size = batch_size_stage_1, epochs = epochs_stage_1,
                        verbose=1, shuffle=False)
  else:
    print('loading model')
    model = tf.keras.models.load_model(load_model_name)
    print('done')

  orig_pred_train, train_true = \
  pred_and_results(train_X, model, train_data_frame, output_len, mean_vec_train, std_vec_train)
  train_rmse_without_adj = np.sqrt(np.mean((orig_pred_train - train_true)**2))
  print('train_rmse_without_adj:', train_rmse_without_adj)

  # detect heteroscedastic series
  diff = train_true / orig_pred_train # multiplicative model
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

  train_true_lst = split_to_series(train_data_frame[:, -output_len], num_window_per_series_train, num_series)
  # test for the entire dataset
  print('start testing')
  test_data_frame, norm_test_data_frame, num_window_per_series_test, mean_vec_test, std_vec_test = \
  split_windows(xtest_lst, num_series, input_len, output_len)
  test_X = norm_test_data_frame[:, :input_len].reshape((-1,input_len,1))
  test_pred, test_true = \
  pred_and_results(test_X, model, test_data_frame, output_len, mean_vec_test, std_vec_test)
  test_pred_lst = split_to_series(test_pred, num_window_per_series_test, num_series)
  test_true_lst = split_to_series(test_true, num_window_per_series_test, num_series)
  test_metrics_without_adj = metrics(num_series, test_true_lst, test_pred_lst)

  # adj model
  print('entering stage 2')
  num_series, test_true_lst, test_pred_with_adj_lst = stage_2_m3(xtrain_lst, xtest_lst, num_series, input_len, output_len,
                                  model, epochs_stage_2, batch_size_stage_2,
                                  pred_lst, train_true_lst, test_pred_lst, test_true_lst,
                                  hetero_series_idx, diff_lst,
                                  seasonal_period, random_seed,
                                  num_window_per_series_train, num_window_per_series_test,
                                  required_num_of_clusters = required_num_of_clusters)
  # metrics_res = {'res_without_adj': test_metrics_without_adj,
  #              'res_with_adj': test_metrics_with_adj}
  # metrics_res_df = pd.DataFrame(metrics_res)
  return([num_series, test_true_lst, test_pred_lst, test_pred_with_adj_lst])

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
run_global_domain_global_m3(m3_1_data, m3_1_info, input_len = 24, output_len = 1, seasonal_period = 12,
                             load_model_name = None, build_model = build_mlp_model,
                             batch_size_stage_1 = 32, batch_size_stage_2 = 4,
                             epochs_stage_1 = 30, epochs_stage_2 = 40,
                             required_num_of_clusters = 3,
                             random_seed = 3407)

metrics_m3(num_series_m3_1, true_demand_lst_m3_1,
           pred_test_demand_lst_without_adj_m3_1, pred_test_demand_lst_with_adj_m3_1)