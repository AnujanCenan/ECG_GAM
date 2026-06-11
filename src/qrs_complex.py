import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv

from fast_qrs_detector import qrs_detector, print_signal_with_qrs


from data_reader.data_reader import CHAPMAN_Reader
from plotter.plotter import Plotter


load_dotenv()
DATA_DIR = os.getenv("CHAPMAN_DATASET")
TIME_SEGMENT = 10

if not DATA_DIR:
    print("DATA_DIR not set in .env file. Exiting...")
    exit(1)

reader = CHAPMAN_Reader(path=DATA_DIR)

p_sig, freq = reader.get_record("WFDBRecords/01/010/JS00004")

print(p_sig.shape)
print(freq)

plt = Plotter()

p_sig = p_sig[:, 0]


qrs_results = qrs_detector(p_sig, freq)

print(qrs_results / freq)

print_signal_with_qrs(p_sig, qrs_predicted=qrs_results)
