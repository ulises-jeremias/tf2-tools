#!/usr/bin/env python

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tf_tools.model_selection import train_test_split_balanced


def store_split(x, y, path, data_dir, mode='w'):
    f = open(path, mode)
    for img, label in zip(x, y):
        img = os.path.normpath(os.path.relpath(img, data_dir))
        f.write("{} {}\n".format(img, label))
    f.close()


def generate_splits(x, y, split, data_dir, splits_dir, dataset, version, train_size, test_size, n_train_per_class, n_test_per_class, seed, balanced=False, **kargs):
    np.random.seed(seed)

    data_dir = data_dir if data_dir else '/tf/data/{}/data'.format(dataset)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    output_dir = os.path.join(splits_dir, split)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not balanced and n_train_per_class <= 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=train_size, test_size=test_size, stratify=y, random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, train_size=0.8, test_size=0.2, stratify=y_train, random_state=seed)
    else:
        print('Split balanced')
        samples_per_class = int(np.round(n_train_per_class * 1.6))
        x_train, x_test, y_train, y_test = train_test_split_balanced(np.array(x), np.array(y), train_size=train_size, test_size=test_size,
                                                                     n_train_per_class=samples_per_class, n_test_per_class=n_test_per_class, n_dim=False)
        x_train, x_val, y_train, y_val = train_test_split_balanced(
            x_train, y_train, train_size=0.8, n_train_per_class=n_train_per_class, test_size=0.2, n_dim=False)

    store_split(x_test, y_test, os.path.join(output_dir, 'test.txt'), data_dir)
    store_split(x_train, y_train, os.path.join(
        output_dir, 'train.txt'), data_dir)
    store_split(x_val, y_val, os.path.join(output_dir, 'val.txt'), data_dir)
    store_split(x_train, y_train, os.path.join(
        output_dir, 'trainval.txt'), data_dir)
    store_split(x_val, y_val, os.path.join(
        output_dir, 'trainval.txt'), data_dir, 'a+')
