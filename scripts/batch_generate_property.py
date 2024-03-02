import pandas as pd
import os
import sys

sys.path.append('..')
from generate_feature import Batch_BasicDes


def GenerateSample(csv_path, save_path):
    all_sample = pd.read_csv(csv_path, index_col=0)
    sequence = all_sample["seq"]
    peptides = sequence.values.copy().tolist()
    save_path = os.path.join(save_path, os.path.split(csv_path)[-1])
    Batch_BasicDes.cal_pep(peptides, sequence, save_path)


if __name__ == "__main__":
    data_path = YOU_DATA_DIR
    save_path = RESULT_SAVE_PATH

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file_name in os.listdir(data_path):
        if os.path.splitext(file_name)[-1] == '.csv':
            print(file_name)
            GenerateSample(csv_path=os.path.join(data_path, file_name), save_path=save_path)
        else:
            continue
