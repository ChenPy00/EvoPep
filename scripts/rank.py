import pandas as pd
import os
import time
from glob import glob

#  *.csv代表这个路径下所有的csv文件
result_path = RESULT_PATH  # 待排序的文件保存的文件夹，即模型评分后的文件所在文件夹，需要先运行screen.py
property_path = PROPERTY_PATH  # 生成理化性质后的文件夹，从这里索引理化性质
save_path = SAVE_PATH  # 排序后的文件保存的文件夹,可以修改
if not os.path.exists(save_path):
    os.makedirs(save_path)

rank_df = []

for i, csv_name in enumerate(glob(result_path)):
    print(i, ' ', csv_name)
    start = time.time()
    result_file = os.path.join(result_path, csv_name)
    property_file = os.path.join(property_path, csv_name)

    result_csv = pd.read_csv(result_file, index_col=0)
    property_csv = pd.read_csv(property_file, index_col=0)

    result_csv['probability'] = result_csv['amp_predict'] + result_csv['cpp_predict']  # 这里采用两种分数之和作为排序的依据
    #  这里筛选两种分数都必须大于0.9，需要的话,把下面这一行最前面的#和result_csv前的空格删掉
    #  这里是先简单筛选一遍，然后再合并,通过简单的筛选，减少数据量，可以减少数据整合的时间，但是并不会减少很多
    #  经过测试，rank CNN和LSTM的文件时，不宜将两个分数设置得过高，最后不要高于0.9，否则会达不到最后的筛选数量
    result_csv = result_csv.query('amp_predict > 0.9 & cpp_predict > 0.9')

    df = result_csv.reset_index().merge(property_csv.reset_index())
    # rank_df = pd.concat([rank_df, df])
    rank_df.append(df)
    stop = time.time()
    print(stop-start,'s')

rank_df = pd.concat(rank_df)
rank_df = rank_df.sort_values('probability', ascending=False)
rank_df[:10000].to_csv(os.path.join(save_path, 'rank_result_10000_230830.csv'))