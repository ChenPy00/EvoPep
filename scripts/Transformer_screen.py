import os
import sys
sys.path.append('..')
from utils import *
from dataset import LoadDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from glob import glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ ['CUDA_VISIBLE_DEVICES'] = '0'

from tqdm import tqdm
import time

for i in tqdm(range(100)):
    time.sleep(0.5)

if __name__ == '__main__':
    setup_seed(0)
    tokenizer, config = get_TokenizerConfig()
    Apmodel = load_model(config, use_property=True, CUDA=True)
    Apmodel.load_state_dict(torch.load(YOU_MODEL_SAVE_PATH))
    cppmodel = load_model(config, use_property=True, CUDA=True)
    cppmodel.load_state_dict(torch.load(YOU_MODEL_SAVE_PATH))

    library_path = '/library_path/*.csv'
    save_path = '/save_path/Transformer'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for csv_name in glob(library_path):
        file = os.path.join(library_path, csv_name)
        print(file)
        data = pd.read_csv(file,index_col=0)
        mydataset = LoadDataset.ScreenDataset(data, tokenizer, use_property=True)
        mydataloader = DataLoader(
                        mydataset,
                        batch_size=128, # 这个参数是指一次对多少条序列进行评分，数值需要根据电脑而定
                        num_workers=0, # 这个参数简单来说指采用多少线程来加载处理数据，数值需要根据电脑而定
                        pin_memory=True,
                        shuffle=False
                    )
        Ap_all_predict = []
        cpp_all_predict = []
        with torch.no_grad():
            Apmodel.eval()
            cppmodel.eval()
            for i, inputs in enumerate(tqdm(mydataloader)):
                Apoutput = Apmodel(input_ids=inputs['input_ids'].cuda(),
                               feature=inputs['feature'].cuda(),
                               attention_mask=inputs['attention_mask'].cuda(),
                               return_dict=None, )
                Ap_all_predict += torch.softmax(Apoutput['cls_logits'],dim=-1)[:,1].cpu().tolist()

                cppoutput = cppmodel(input_ids=inputs['input_ids'].cuda(),
                                   feature=inputs['feature'].cuda(),
                                   attention_mask=inputs['attention_mask'].cuda(),
                                   return_dict=None, )
                cpp_all_predict += torch.softmax(cppoutput['cls_logits'], dim=-1)[:, 1].cpu().tolist()

        data['amp_predict'] = Ap_all_predict
        data['cpp_predict'] = cpp_all_predict
        data[['seq','amp_predict','cpp_predict']].to_csv(os.path.join(save_path, os.path.split(file)[-1]))
