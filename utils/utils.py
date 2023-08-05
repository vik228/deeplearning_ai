import random
import math
import numpy as np
import pandas as pd
from dsl.settings import DATASET_BASE_PATH
from datetime import datetime


def get_data_path(dir=None):
    if dir:
        data_dir = "{0}/{1}".format(DATASET_BASE_PATH, dir)
    else:
        data_dir = "{0}".format(DATASET_BASE_PATH)
    train_dataset_path = "{0}/train.csv".format(data_dir)
    test_dataset_path = "{0}/test.csv".format(data_dir)
    return train_dataset_path, test_dataset_path


def load_data_from_csv_as_np(label_idx, dir=None, **kwargs):
    train_dataset_path, test_dataset_path = get_data_path(dir)
    train_data = np.genfromtxt(train_dataset_path, delimiter=",", **kwargs)
    test_data = np.genfromtxt(test_dataset_path, delimiter=",", **kwargs)
    train_data_X = train_data[1:, label_idx:]
    train_data_Y = train_data[1:, :label_idx]
    return train_data_X, train_data_Y, test_data[1:]


def load_data_from_csv_as_pd(label_key, dir=None, **kwargs):
    train_dataset_path, test_dataset_path = get_data_path(dir)
    train_data = pd.read_csv(train_dataset_path, **kwargs)
    test_data = pd.read_csv(test_dataset_path)
    train_data_y = train_data.loc[:, [label_key]]
    train_data_x = train_data.loc[:, train_data.columns != label_key]
    return train_data_x, train_data_y, test_data


def read_csv_in_pd(dir, filename):
    if dir:
        data_dir = "{0}/{1}/{2}".format(DATASET_BASE_PATH, dir, filename)
    else:
        data_dir = "{0}/{1}".format(DATASET_BASE_PATH, filename)
    return pd.read_csv(data_dir)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    np.random.seed(seed)
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_Y = X[permutation]
    mini_batches = []
    number_of_mini_batches = math.floor(m / mini_batch_size)
    for b in range(0, number_of_mini_batches):
        mini_batch_X = shuffled_X[b * mini_batch_size : (b + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[b * mini_batch_size : (b + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[number_of_mini_batches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[number_of_mini_batches * mini_batch_size : m]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    return mini_batches


def merge_nested_dicts(dict1, dict2):
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(merge_nested_dicts(dict1[k], dict2[k])))
            else:
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])


def get_or_drop_na_cols(df, **kwargs):
    threshold = kwargs.get("threshold", 0.0)
    drop = kwargs.get("drop", False)
    nan_cols = [col for col in df.columns if df[col].isnull().mean() >= threshold]
    if drop:
        df.drop(nan_cols, axis=1, inplace=True)
    return nan_cols


def get_cols_with_nan_values(df):
    return [col for col in df.columns if df[col].isnull().any()]


def replace_categorical_nan_cols(df, columns, **kwargs):
    replace_func = kwargs.get("replace_func")
    if replace_func == "mode":
        for col in columns:
            df[col] = df[col].fillna(df[col].mode()[0])


def replace_numerical_nan_cols(df, columns, **kwargs):
    replace_func = kwargs.get("replace_func")
    if replace_func == "mean":
        for col in columns:
            df[col] = df[col].fillna(df[col].mean())


def convert_categorical_to_one_hot_encoding(df, columns, **kwargs):
    cleaned_df = df
    for col in columns:
        dummy_df = pd.get_dummies(cleaned_df[col], drop_first=True)
        cleaned_df.drop(col, axis=1, inplace=True)
        cleaned_df = pd.concat([cleaned_df, dummy_df], axis=1)
    return cleaned_df


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            "\n Time taken: %i hours %i minutes and %s seconds. "
            % (thour, tmin, round(tsec, 2))
        )


def train_test_split(df, test_size):
    random.seed(0)
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df
