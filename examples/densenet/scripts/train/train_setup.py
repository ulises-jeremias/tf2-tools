"""
Logic for model creation, training launching and actions needed to be
accomplished during training (metrics monitor, model saving etc.)
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
import tf_tools.engines.train as train_engine
from datetime import datetime
from densenet import densenet_model
from examples.datasets import load
from tf_tools.weighted_loss import weighted_loss


def train(config):
    np.random.seed(2020)
    tf.random.set_seed(2020)

    # Useful data
    now = datetime.now()
    now_as_str = now.strftime('%y_%m_%d-%H:%M:%S')

    # Output files
    checkpoint_path = f"{config['model.save_path']}"
    config_path = f"{config['output.config_path'].format(now_as_str)}"
    csv_output_path = f"{config['output.train_path'].format(now_as_str)}"
    train_summary_file_path = f"{config['summary.save_path'].format('train', config['data.dataset'], config['model.name'], config['model.type'], now_as_str)}"
    test_summary_file_path = f"{config['summary.save_path'].format('test', config['data.dataset'], config['model.name'], config['model.type'], now_as_str)}"
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
    file.write(json.dumps(config, indent=2))
    file.close()

    # create summary file if not exists
    if not os.path.exists(summary_path):
        file = open(summary_path, 'w')
        file.write("datetime, model, config, min_loss, min_loss_accuracy\n")
        file.close()

    # Data loader
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    train, val, _, nb_classes, image_shape, class_weights = load(
        config, datagen_flow=True)

    (train_gen, train_size, _) = train
    (val_gen, val_size, _) = val

    # Determine device
    if config['data.cuda']:
        cuda_num = config['data.gpu']
        device_name = f'GPU:{cuda_num}'
    else:
        device_name = 'CPU:0'

    if config['data.weight_classes']:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        loss_object = weighted_loss(loss_object, class_weights)
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam()

    time_start = time.time()
    # Compiles a model, prints the model summary, and saves the model diagram into a png file.
    model = densenet_model(classes=nb_classes, shape=image_shape,
                           growth_rate=config['model.growth_rate'], nb_layers=config['model.nb_layers'], reduction=config['model.reduction'])
    # model.summary()

    # tf.keras.utils.plot_model(model, "{}/model.png".format(results_dir), show_shapes=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='val_accuracy')

    # create summary writers
    train_summary_writer = tf.summary.create_file_writer(
        train_summary_file_path)
    val_summary_writer = tf.summary.create_file_writer(test_summary_file_path)

    print("Starting training")

    with tf.device(device_name):
        loss, acc = train_engine.train(
            model=model, batch_size=config['data.batch_size'],
            epochs=config['train.epochs'], max_patience=config['train.patience'],
            train_gen=train_gen, train_size=train_size, val_gen=val_gen, val_size=val_size,
            train_loss=train_loss, train_accuracy=train_accuracy,
            test_loss=val_loss, test_accuracy=val_accuracy,
            val_loss=val_loss, val_accuracy=val_accuracy,
            optimizer=optimizer, loss_object=loss_object,
            checkpoint_path=checkpoint_path,
            train_summary_writer=train_summary_writer,
            val_summary_writer=val_summary_writer,
            csv_output_file=csv_output_path,
            nb_classes=nb_classes,
            engine=config['engine'], lr=config['train.lr'],
        )

    time_end = time.time()

    summary = "{}, {}, {}, {}, {}, {}\n".format(
        now_as_str, config['data.dataset'], config['model.name'], config_path, loss, acc)
    print(summary)

    file = open(summary_path, 'a+')
    file.write(summary)
    file.close()

    elapsed = time_end - time_start
    h, min = elapsed//3600, elapsed % 3600//60
    sec = elapsed-min*60

    print(f"Training took: {h:.2f}h {min:.2f}m {sec:.2f}s!")