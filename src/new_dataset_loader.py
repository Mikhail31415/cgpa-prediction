from pathlib import Path

import numpy as np
import pandas as pd
import torch

DATA_DIR = Path(r'C:\Python\projects\CGPAPrediction\data\final_grade_prediction_dataset')


def group_other_semesters(df):
    mask = df['Semester'] == 4
    last_semester = df[mask].copy()
    previous_semesters = df[~mask]

    last_semester['Other semesters'] = [None] * len(last_semester)

    for idx, student_df in previous_semesters.groupby('Student ID', as_index=False):
        by_semester = student_df.drop('Student ID', axis=1).groupby('Semester')
        semesters = [None] * len(by_semester)
        for i, semester_data in by_semester:
            semesters[i] = semester_data.drop('Semester', axis=1)

        mask_x = (last_semester['Student ID'] == idx)
        for i in last_semester.loc[mask_x].index:
            last_semester.at[i, 'Other semesters'] = semesters

    return last_semester


def grade_to_cat(x):
    if x < 70:
        return 0
    elif x < 90:
        return 1
    return 2


def calc_embedding_dim(i):
    return min(50, int(i ** 0.5))


def to_one_hot(arr: np.array, max_size):
    res = np.zeros((arr.size, max_size), dtype='float32')
    res[np.arange(arr.size), arr] = 1.0
    return res


embedding_cols = {'Educational Program / Major': 'educational_program', 'Course': 'course',
                  'Instructor Full Name': 'instructor_name', 'Instructor Department': 'instructor_department'}
one_hot_cols = ['Payment Type', 'Funding Type', 'Language of Instruction', 'Institute', 'Study Mode',
                'Student Gender', 'Instructor Gender']
number_cols = ['Midterm 1', 'Midterm 2', 'Final Grade', 'Rating']

features = ['educational_program', 'course', 'instructor_name', 'instructor_department', 'tensor_features']
other_semester_features = ['educational_program', 'course', 'instructor_name', 'instructor_department',
                           'tensor_features', 'mask']

def pad_to(x, target_len, dim=0):
    pad_size = target_len - x.size(dim)
    if pad_size <= 0:
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_size

    pad = torch.zeros(*pad_shape, dtype=x.dtype)
    padded_x = torch.cat([x, pad], dim=dim)

    return padded_x


def to_torch_tensors(df, one_hot_dims, padmin=10):
    res = {}

    if len(df) <= padmin:
        mask = torch.zeros(padmin, dtype=torch.bool)
        mask[len(df):] = True
        res['mask'] = mask

    for col_name, arg_name in embedding_cols.items():
        res[arg_name] = pad_to(torch.from_numpy(df[col_name].values.astype('int64')), padmin)

    to_cat = []
    for col in one_hot_cols:
        to_cat.append(pad_to(torch.from_numpy(to_one_hot(df[col].values.astype('int64'), one_hot_dims[col])), padmin))

    for col in number_cols:
        if col in df.columns:
            to_cat.append(pad_to(torch.from_numpy(df[col].values).unsqueeze(1), padmin))

    res['tensor_features'] = torch.cat(to_cat, dim=1)

    return res


def concat_dicts(l, ks):
    res = {}
    for k in ks:
        res[k] = torch.stack(list(d[k] for d in l))
    return res


def get_np_dataset(embedding_threshold):
    train_df, val_df, test_df = pd.read_csv(DATA_DIR / 'train.csv'), pd.read_csv(DATA_DIR / 'val.csv'), pd.read_csv(
        DATA_DIR / 'test.csv')
    data = {'train': train_df, 'val': val_df, 'test': test_df}

    for df in data.values():
        df.drop(['Exam', 'Group'], axis=1, inplace=True)

    for col in number_cols:
        for df in data.values():
            df[col] = df[col].astype('float32')

    for df in data.values():
        df[['Midterm 1', 'Midterm 2', 'Rating']] = df[['Midterm 1', 'Midterm 2', 'Rating']].apply(lambda x: x / 100)

    for df in data.values():
        mask = df['Semester'] != 4
        df.loc[mask, 'Final Grade'] /= 100

    embedding_dims = {}
    for col in embedding_cols.keys():
        input_shape = len([v for v in train_df[col].value_counts().values if v >= embedding_threshold]) + 1
        embedding_dims[col] = (input_shape, calc_embedding_dim(input_shape))

    for df in data.values():
        df['hs'] = df['Course']

    for col, dim in embedding_dims.items():
        max_index = dim[0] - 2
        unk_index = max_index + 1
        for df in data.values():
            df.loc[:, col] = df[col].apply(lambda i: unk_index if i > max_index else i)

    one_hot_dims = {}
    for col in one_hot_cols:
        one_hot_dims[col] = len(train_df[col].unique())

    y_data = {}
    for k, df in data.items():
        data[k] = group_other_semesters(df)
        y_data[k] = data[k]['Final Grade']
        data[k] = data[k].drop(['Final Grade'], axis=1)

    other_semesters = {k: df['Other semesters'] for k, df in data.items()}

    for k, df in data.items():
        data[k] = to_torch_tensors(df, one_hot_dims)

    for k, osem in other_semesters.items():
        other_semesters = []
        for student_semesters in osem:
            student_semesters_converted = []
            for semester in student_semesters:
                student_semesters_converted.append(to_torch_tensors(semester, one_hot_dims))
            other_semesters.append(concat_dicts(student_semesters_converted, other_semester_features))

        data[k]['other_semesters'] = concat_dicts(other_semesters, other_semester_features)


    for k, s in y_data.items():
        y_data[k] = torch.from_numpy(s.apply(grade_to_cat).astype('int64').values)

    return data, y_data, embedding_dims, one_hot_dims
