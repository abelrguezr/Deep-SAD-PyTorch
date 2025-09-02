# Deep SVDD for Network Intrusion Detection Systems

This repository contains a modified version of the Deep SAD PyTorch implementation, adapted for Network Intrusion Detection Systems (NIDS) research, using the [PyTorch](https://pytorch.org/) implementation of the _Deep SAD_ method presented in the ICLR 2020 paper ”Deep Semi-Supervised Anomaly Detection”.

## Overview

This work was developed as part of my master's thesis in 2020, focusing on Network Intrusion Detection Systems (NIDS), specifically addressing anomaly detection in non-stationary settings using real network traffic. The core idea was to train to map benign traffic into a minimal volume hypersphere in the learned feature space, and mark as anomalous traffic lying outside those boundaries. I used Ray and Ax frameworks for distributed experiment scheduling and hyperparameter tuning respectively.

### Key changes

- **Distributed Computing**: Utilizes [Ray](https://ray.io/) for scalable distributed experiment execution
- **Hyperparameter Optimization**: Implements [Ax](https://ax.dev/) for hyperparameter tuning and experiment management

- **Network Traffic Processing**: Custom data loaders and preprocessing pipelines for handling real network traffic data (CICFLOW, MAWILab, NSL-KDD)

## Installation

This code is written in `Python 3.7+` and requires the packages listed in `requirements.txt`.

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### `virtualenv`

```bash
cd <path-to-Deep-SAD-PyTorch-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`

```bash
cd <path-to-Deep-SAD-PyTorch-directory>
conda create --name myenv python=3.7
conda activate myenv
pip install -r requirements.txt
```

### Additional Dependencies

For distributed computing and hyperparameter optimization:

```bash
pip install ray[tune] ax-platform
```

## License

MIT
