import random
import os
import pandas as pd
import numba
import numpy as np
import time
from tqdm import tqdm




alphabet = 'ACDEFGHIKLMNPQRSTVWY'
alphabet_list = [a for a in alphabet]

library_path = r'E:\LS\深度学习\code\practice\14肽\生成14肽库'  # 保存生成肽库后的文件的文件夹,需要修改
subset_num = 20000  # 500的意思是分500份文件保存
total_num = 1e9  # 1e9是10的9次方
experience_screen_path = r'E:\LS\深度学习\code\practice\14肽\经验筛选'  # 通过经验筛选肽库后的文件的文件夹,需要修改

############### 生成肽库 ###############
library = []
for i in range(int(1e9)):
    if (i + 1) % subset_num == 0:
        library = pd.DataFrame(data=library, columns=['seq'])
        library.to_csv(os.path.join(library_path, f'random_library_{int((i + 1) / subset_num)}.csv'))
        library = []
    #num = random.randint(10, 20)  # 长度选择，这里是为10-20,不定长度用这个，把前面的#去掉就可以用了
    num = 14  # 长度选择，这里是为16，固定长度用这个，前面加个#可以忽略了，和上面的不定长度不可同时使用
    library.append(''.join(random.choices(alphabet_list, k=num)))

############### 经验筛选 ###############
@numba.jit(nopython=True)
def judging_experience(seq):
    hydrophilic_count = 0
    hydrophobe_count = 0
    hydrophilic = ['R', 'N', 'D', 'E', 'Q', 'K', 'H', 'S', 'T']
    hydrophobe = ["I", "L", "A", "G", "M", "F", "W", "V"]
    for element in seq:
        if element in hydrophobe:
            hydrophobe_count += 1
        elif element in hydrophilic:
            hydrophilic_count += 1

    hydrophobe_percent = (hydrophobe_count / len(seq))
    hydrophilic_percent = (hydrophilic_count / len(seq))
    amphiphilicity = (0.3 < hydrophobe_percent < 0.7) and (0.3 < hydrophilic_percent < 0.7)
    positive_charge = (seq.count('K') + seq.count('R') + seq.count('H') - seq.count('D') - seq.count('E') > 0)
    R_proportion = (seq.count('R') / len(seq)) < 0.25
    return amphiphilicity and positive_charge and R_proportion


for i in range(int(1e9/subset_num)):
    start = time.time()
    data = pd.read_csv(os.path.join(library_path, f'random_library_{i + 1}.csv'),index_col=0)
    _bool = data.seq.apply(judging_experience)
    data[_bool].to_csv(os.path.join(experience_screen_path, f'random_library_{i + 1}.csv'))
    stop = time.time()
    print('第{}个文件 使用{}s筛选'.format(i+1, stop-start))

