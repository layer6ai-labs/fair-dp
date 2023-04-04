import os
import random as rand

import pandas as pd
import regex as re
import torch

from .dataset import GroupLabelDataset
from .sample_weights import find_sample_weights


# normalize df columns
def normalize(df, columns):
    result = df.copy()
    for column in columns:
        mu = df[column].mean(axis=0)
        sigma = df[column].std(axis=0)
        assert sigma != 0
        result[column] = (df[column] - mu) / sigma
    return result


def make_tabular_train_valid_split(data, valid_frac):
    n_valid = int(valid_frac * data.shape[0])
    valid_data = data[:n_valid]
    train_data = data[n_valid:]
    return train_data, valid_data


def make_tabular_train_valid_test_split(data, valid_frac, test_frac, seed):
    # shuffle samples
    data = data.sample(frac=1, random_state=seed)

    n_test = int(test_frac * data.shape[0])
    test_data = data[:n_test]
    data = data[n_test:]

    train_data, valid_data = make_tabular_train_valid_split(data, valid_frac)

    return train_data, valid_data, test_data


# refer to sample_weights.py
def sample_by_group_ratios(group_ratios, df, seed):
    print("Number of samples by group (before sampling):")
    print(df.protected_group.value_counts())
    sample_weights = find_sample_weights(group_ratios, df.protected_group.value_counts().tolist())
    rand.seed(seed)
    idx = [rand.random() <= sample_weights[row.protected_group] for _, row in df.iterrows()]
    df = df.loc[idx]
    print("Number of samples by group (after sampling):")
    print(df.protected_group.value_counts())
    return df


def preprocess_adult(df, protected_group, target, group_ratios, seed):
    numerical_columns = ["age", "education_num", "capital_gain", "capital_loss",
                         "hours_per_week"]
    if protected_group in numerical_columns:
        numerical_columns.remove(protected_group)
    df = normalize(df, numerical_columns)

    mapped_income_values = df.income.map({"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1})
    df.loc[:, "income"] = mapped_income_values

    mapped_sex_values = df.sex.map({"Male": 0, "Female": 1})
    df.loc[:, "sex"] = mapped_sex_values

    # make race binary
    def race_map(value):
        if value != "White":
            return (1)
        return (0)

    mapped_race_values = df.race.map(race_map)
    df.loc[:, "race"] = mapped_race_values

    categorical = df.columns.tolist()
    for column in numerical_columns:
        categorical.remove(column)
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

    df = sample_by_group_ratios(group_ratios, df, seed)

    # convert to one-hot vectors
    categorical_non_binary = ["workclass", "education", "marital_status", "occupation",
                              "relationship", "native_country"]
    df = pd.get_dummies(df, columns=categorical_non_binary)

    return df


def preprocess_dutch(df, protected_group, target, group_ratios, seed):
    # remove weight feature
    df = df.drop("weight", axis=1)

    # drop underage samples (under 14 yrs old)
    df = df.drop(df[df.age <= 3].index)

    # drop samples with occupation = not working, unknown = 999, 998
    df = df.drop(df[df.occupation == 999].index)
    df = df.drop(df[df.occupation == 998].index)

    # map occupations to high=1, mid, low=0
    occupation_map = {
        1: 1,
        2: 1,
        3: "mid",
        4: 0,
        5: 0,
        6: "mid",
        7: "mid",
        8: "mid",
        9: 0
    }
    mapped_occupation_values = df.occupation.map(occupation_map)
    df.loc[:, "occupation"] = mapped_occupation_values

    # drop samples with occupation = mid
    df = df.drop(df[df.occupation == "mid"].index)

    mapped_sex_values = df.sex.map({1: 0, 2: 1})
    df.loc[:, "sex"] = mapped_sex_values

    # note original dataset has values {0,1,9} for prev_res_place, but all samples with 9 are underage, hence get dropped 
    mapped_prev_res_place_values = df.prev_res_place.map({1: 0, 2: 1})
    df.loc[:, "prev_res_place"] = mapped_prev_res_place_values

    categorical = df.columns.to_list()
    print("Possible protected groups are: {}".format(categorical))

    if protected_group == "labels":
        df.loc[:, "protected_group"] = df[target]
    elif protected_group not in categorical:
        raise ValueError(
            f"Invalid protected group {protected_group}. "
            + f"Valid choices are {categorical}."
        )
    else:
        df.loc[:, "protected_group"] = df[protected_group]

        # convert categorical unprotected features to one-hot vectors
    if target in categorical:
        categorical.remove(target)
    if "sex" in categorical:
        categorical.remove("sex")  # binary
    if "prev_res_place" in categorical:
        categorical.remove("prev_res_place")  # binary

    df = sample_by_group_ratios(group_ratios, df, seed)

    df = pd.get_dummies(df, columns=categorical)

    return df


def get_dutch_raw(data_root, valid_frac, test_frac, seed, protected_group, target, group_ratios):
    '''
    Dutch dataset:
    Download from https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:32357 (free registration required)
    unzip and save directory to fair-dp/data/dutch/
    '''

    columns = ["sex", "age", "household_posn", "household_size", "prev_res_place", "citizenship",
               "country_birth", "edu_level", "econ_status", "occupation", "cur_eco_activity",
               "marital_status", "weight"]
    col_str = ",".join(columns)
    col_str = col_str + "\n"

    # data location data/dutch/original/org/IPUMS2001.asc
    write_from = open(os.path.join(data_root, "dutch", "original", "org", "IPUMS2001.asc"), "r")
    write_to = open(os.path.join(data_root, "dutch", "dutch_data_formatted.csv"), "w")

    write_to.write(col_str)

    def to_csv(write_from, write_to):
        while True:
            line = write_from.readline()

            if not line:
                break

            # refer to IMPUS2001_meta.pdf page 8 for group values
            result = re.search(r"(.{1})(.{2})(.{4})(.{3})(.{3})(.{2})(.{1})(.{2})(.{3})(.{3})(.{3})(.{1})(.{16})", line)
            formatted_str = ""
            for group in range(1, len(result.groups()) + 1):
                if group == len(result.groups()):
                    formatted_str = formatted_str + result.group(group).strip() + "\n"
                else:
                    formatted_str = formatted_str + result.group(group).strip() + ","
            write_to.write(formatted_str)

    to_csv(write_from, write_to)
    write_from.close()
    write_to.close()

    df = pd.read_csv(os.path.join(data_root, "dutch", "dutch_data_formatted.csv"))

    df_preprocessed = preprocess_dutch(df, protected_group, target, group_ratios, seed)

    train_raw, valid_raw, test_raw = make_tabular_train_valid_test_split(df_preprocessed, valid_frac, test_frac, seed)

    return train_raw, valid_raw, test_raw


def get_adult_raw(data_root, valid_frac, test_frac, seed, protected_group, target, group_ratios):
    '''
    Adult dataset:
    Download from https://archive.ics.uci.edu/ml/datasets/Adult
    and save files adult.data, adult.test to fair-dp/data/adult/
    '''
    columns = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
               "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
               "hours_per_week", "native_country", "income"]

    df_1 = pd.read_csv(os.path.join(data_root, "adult", "adult.data"), sep=", ", engine='python', header=None)
    df_2 = pd.read_csv(os.path.join(data_root, "adult", "adult.test"), sep=", ", engine='python', header=None,
                       skiprows=1)
    df_1.columns = columns
    df_2.columns = columns
    df = pd.concat((df_1, df_2), ignore_index=True)

    df = df.drop("fnlwgt", axis=1)
    for column in df.columns:
        df = df[df[column] != "?"]
    df.to_csv(os.path.join(data_root, "adult", "adult_data_formatted.csv"), index=False)

    df = pd.read_csv(os.path.join(data_root, "adult", "adult_data_formatted.csv"))

    df_preprocessed = preprocess_adult(df, protected_group, target, group_ratios, seed)

    train_raw, valid_raw, test_raw = make_tabular_train_valid_test_split(df_preprocessed, valid_frac, test_frac, seed)

    return train_raw, valid_raw, test_raw


def get_tabular_datasets(name, data_root, seed, protected_group, group_ratios=None, make_valid_loader=False):
    if name == "adult":
        data_fn = get_adult_raw
        target = "income"
    elif name == "dutch":
        data_fn = get_dutch_raw
        target = "occupation"
    else:
        raise ValueError(f"Unknown dataset {name}")

    valid_frac = 0
    if make_valid_loader:
        valid_frac = 0.1
    test_frac = 0.2
    train_raw, valid_raw, test_raw = data_fn(data_root, valid_frac, test_frac, seed, protected_group, target,
                                             group_ratios)

    feature_columns = train_raw.columns.to_list()
    feature_columns.remove(target)
    feature_columns.remove("protected_group")

    train_dset = GroupLabelDataset("train",
                                   torch.tensor(train_raw[feature_columns].values, dtype=torch.get_default_dtype()),
                                   torch.tensor(train_raw[target].to_list(), dtype=torch.long),
                                   torch.tensor(train_raw["protected_group"].values.tolist(), dtype=torch.long)
                                   )
    valid_dset = GroupLabelDataset("valid",
                                   torch.tensor(valid_raw[feature_columns].values, dtype=torch.get_default_dtype()),
                                   torch.tensor(valid_raw[target].to_list(), dtype=torch.long),
                                   torch.tensor(valid_raw["protected_group"].values.tolist(), dtype=torch.long)
                                   )
    test_dset = GroupLabelDataset("test",
                                  torch.tensor(test_raw[feature_columns].values, dtype=torch.get_default_dtype()),
                                  torch.tensor(test_raw[target].to_list(), dtype=torch.long),
                                  torch.tensor(test_raw["protected_group"].values.tolist(), dtype=torch.long)
                                  )

    return train_dset, valid_dset, test_dset
