import click
import os
import pandas as pd
import torch
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from skmultiflow.drift_detection.hddm_w import HDDM_W
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from datasets.cicflow import CICFlowADDataset
from models.deepSVDD import DeepSVDD
from datasets.main import load_dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():
    def eval_drift(detector, element):
        detector.add_element(element)

        return detector.detected_warning_zone(), detector.detected_change()

    predictions = np.random.randint(2, size=2000)
    predictions[1500:]=np.random.randint(1000, size=500)
    
    hddm_w = HDDM_W()
    hddm_a = HDDM_A()
    eddm = EDDM()

    hddm_w_warn, hddm_w_change = zip(
        *[eval_drift(hddm_w, e) for e in predictions])
    hddm_a_warn, hddm_a_change = zip(
        *[eval_drift(hddm_a, e) for e in predictions])
    eddm_warn, eddm_change = zip(
        *[eval_drift(eddm, e) for e in predictions])

    plt.figure()
    plt.plot(hddm_w_change, label='hddm-w')
    plt.plot(hddm_a_change, label='hddm-a')
    plt.plot(eddm_change, label='eddm')
    # plt.plot(hddm_w_warn, label='hddm-w')
    # plt.plot(hddm_a_warn, label='hddm-a')
    # plt.plot(eddm_warn, label='eddm')
    plt.legend(loc=0)
    
    plt.figure()
    plt.plot(predictions)

    plt.show()



if __name__ == '__main__':
    main()