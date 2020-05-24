import os
import json
import numpy as np
import tensorflow as tf

from datetime import datetime
from tf_tools.engines.steps import steps
from .train import train as normal_train
from .maml import train as maml_train


def train(*args, log_info=None, **kwargs):
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%y_%m_%d-%H:%M:%S')

    # Output files
    checkpoint_path = f"{log_info['model.save_path']}"
    config_path = f"{log_info['output.config_path'].format(now_as_str)}"
    csv_output_path = f"{log_info['output.train_path'].format(now_as_str)}"
    train_summary_file_path = f"{log_info['summary.save_path'].format('train', log_info['data.dataset'], log_info['model.name'], log_info['model.type'], now_as_str)}"
    test_summary_file_path = f"{log_info['summary.save_path'].format('test', log_info['data.dataset'], log_info['model.name'], log_info['model.type'], now_as_str)}"
    summary_path = f"results/summary.csv"

    # Output dirs
    data_dir = f"data/"
    checkpoint_dir = checkpoint_path[:checkpoint_path.rfind('/')]
    config_dir = config_path[:config_path.rfind('/')]
    results_dir = csv_output_path[:csv_output_path.rfind('/')]

    # Create folder for model
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create output for train process
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file = open(f"{csv_output_path}", 'w')
    file.write("")
    file.close()

    # Create folder for config
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # generate config file
    file = open(config_path, 'w')
    file.write(json.dumps(log_info, indent=2))
    file.close()

    # create summary file if not exists
    if not os.path.exists(summary_path):
        file = open(summary_path, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
        file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # create summary writers
    train_summary_writer = tf.summary.create_file_writer(
        train_summary_file_path)
    val_summary_writer = tf.summary.create_file_writer(test_summary_file_path)

        
    (train_step, meta_step, test_step) = steps(*args, **kwargs)

    kwargs['train_step'] = kwargs.get('train_step', train_step)
    kwargs['meta_step'] = kwargs.get('meta_step', meta_step)
    kwargs['test_step'] = kwargs.get('test_step', test_step)

    train_engine = maml_train if kwargs.get('engine') == 'maml' else normal_train

    print("Starting training")

    kwargs['train_summary_writer'] = train_summary_writer,
    kwargs['val_summary_writer'] = val_summary_writer
    kwargs['csv_output_file'] = csv_output_path

    loss, acc = train_engine(*args, **kwargs)

    time_end = time.time()

    summary = "{}, {}, {}, {}, {}, {}\n".format(
        now_as_str, log_info['data.dataset'], log_info['model.name'], config_path, loss, acc)
    print(summary)

    file = open(summary_path, 'a+')
    file.write(summary)
    file.close()

    return loss, acc

