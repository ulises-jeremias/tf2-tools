# TensorFlow Tools

Scripts and utilities for training and testing models, datasets splitting and more! All written in TensorFlow v2.

## Content

- [Docs](#docs)
- [Real World Examples](#usage-examples)
    - [Quickstart](#quickstart)
    - [Datasets](#datasets)
    - [Models and Techniques](#models-&-techniques)
        - [Dense Net](#dense-net)
        - [Transfer Learning](#transfer-learning)
    - [Results](#results)
- [Contributors](#contributors)

# Docs

> TODO

* * *

# Real World Examples

In the `examples/` dir there are a lot of real world applications of `tf-tools`. The following subsections explain how to run the examples and observe the results.

## Quickstart

To start the docker container execute the following command

```sh
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build] [-d] [-c <command>]
```

### Tags

- **latest**	The latest release of TensorFlow CPU binary image. Default.
- **nightly**	Nightly builds of the TensorFlow image. (unstable)
version	Specify the version of the TensorFlow binary image, for example: 2.1.0
- **devel**	Nightly builds of a TensorFlow master development environment. Includes TensorFlow source code.

### Variants

> Each base tag has variants that add or change functionality:

- **\<tag\>-gpu**	The specified tag release with GPU support. (See below)
- **\<tag\>-py3**	The specified tag release with Python 3 support.
- **\<tag\>-jupyter**	The specified tag release with Jupyter (includes TensorFlow tutorial notebooks)

You can use multiple variants at once. For example, the following downloads TensorFlow release images to your machine. For example:

```sh
$ ./bin/start -n my-container --build  # latest stable release
$ ./bin/start -n my-container --build -t devel-gpu # nightly dev release w/ GPU support
$ ./bin/start -n my-container --build -t latest-gpu-jupyter # latest release w/ GPU support and Jupyter
```

You can execute

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```
to access the running container's shell.

## Datasets

We used `Cifar10`, `Cifar100` and `MNIST` to create the examples.

## Models & Techniques

### Dense Net

We implemented Densenet using squeeze and excitation layers in tensorflow 2 for our example. To see its implementation go to [densenet](https://github.com/okason97/DenseNet-Tensorflow2).

For more information about densenet please refer to the [original paper](https://arxiv.org/abs/1608.06993).

<details><summary>Training and Eval</summary>

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/run --model densenet --mode train --config <config>
```

`<config> = cifar10 | cifar100 | mnist`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/run --model densenet --mode eval --config <config>
```

`<config> = cifar10 | cifar100 | mnist`
</details>

### Transfer Learning

In this example we can see an implementation of Transfer Learning Technique using the following models
as base: `VGG16`, `VGG19`, `Inception_v3`, `DenseNet`, `DenseNet169`, `DenseNet201`.

<details><summary>Training and Eval</summary>

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/run --tl --model <model> --mode train --config <config>
```

```
<model> = VGG16 | VGG19 | Inception_v3 | DenseNet | DenseNet169 | DenseNet201
<config> = cifar10 | cifar100 | mnist
```
#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/run --tl --model <model> --mode eval --config <config>
```

```
<model> = VGG16 | VGG19 | Inception_v3 | DenseNet | DenseNet169 | DenseNet201
<config> = cifar10 | cifar100 | mnist
```
</details>

## Results

In the `/results` directory you can find the results of a training processes using a `<model>` on a specific `<dataset>`:

```
.
├─ . . .
├─ results
│  ├─ <dataset>                            # results for an specific dataset.
│  │  ├─ <model>                           # results training a <model> on a <dataset>.
│  │  │  ├─ models                         # ".h5" files for trained models.
│  │  │  ├─ results                        # ".csv" files with the different metrics for each training period.
│  │  │  ├─ summaries                      # tensorboard summaries.
│  │  │  ├─ config                         # optional configuration files.
│  │  └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and 
summaries are listed by date.
│  └─ summary.csv                          # contains the summary of all the training
└─ . . .
```

where

```
<dataset> = cifar10 | cifar100 | mnist
<model> = densenet | protonet | VGG16 | VGG19 | Inception_v3 | DenseNet | DenseNet169 | DenseNet201
```

To run TensorBoard, use the following command:

```sh
$ tensorboard --logdir=./results/<dataset>/<model>/summaries
```

* * *

## Contributors

- [Ulises Jeremias Cornejo Fandos](https://github.com/ulises-jeremias)
- [Gastón Gustavo Rios](https://github.com/okason97)
