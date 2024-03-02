import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob

import sys

sys.path.append('..')
from model import CnnNet
# from utils import *
from dataset import LoadDataset
from dataset import DataCollator


if __name__ == '__main__':
    AMP_model = torch.load(YOU_MODEL_SAVE_PATH)
    CPP_model = torch.load(YOU_MODEL_SAVE_PATH)
    AMP_model = AMP_model.cuda()
    CPP_model = CPP_model.cuda()

    library_path = '/library_path/*.csv'
    save_path = '/save_path/CNN'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for csv_name in glob(library_path):
        file = os.path.join(library_path, csv_name)
        print(file)
        data = pd.read_csv(file, index_col=0)
        mydataset = LoadDataset.cnn_dataset(data, use_property=True)
        mydataloader = DataLoader(
            mydataset,
            batch_size=256,  # 这个参数是指一次对多少条序列进行评分，数值需要根据电脑而定，
            num_workers=8,  # 这个参数简单来说指采用多少线程来加载处理数据，数值需要根据电脑而定
            pin_memory=True,
            shuffle=False,
            collate_fn=DataCollator.cnn_collate_fn,
        )
        Ap_all_predict = []
        cpp_all_predict = []
        with torch.no_grad():
            AMP_model.eval()
            CPP_model.eval()
            for i, (x, feature, y) in enumerate(tqdm(mydataloader)):
                Apoutput = AMP_model(x.cuda(), feature.cuda())
                Ap_all_predict += torch.softmax(Apoutput, dim=-1)[:, 1].cpu().tolist()

                cppoutput = CPP_model(x.cuda(), feature.cuda())
                cpp_all_predict += torch.softmax(cppoutput, dim=-1)[:, 1].cpu().tolist()

        data['amp_predict'] = Ap_all_predict
        data['cpp_predict'] = cpp_all_predict
        data[['seq','amp_predict','cpp_predict']].to_csv(os.path.join(save_path, os.path.split(file)[-1]))
