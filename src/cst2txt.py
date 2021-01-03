import pandas as pd
import numpy as np
import os

dataframe = pd.read_csv("../data/rumor/test_data.csv", encoding="utf-8")

target = dataframe['target']
text = dataframe['text']

all_list = []
all_data_path = "../data/rumor/all_data.txt"
data_list_path = "../data/rumor/"

for index, row in dataframe.iterrows():
    all_list.append(str(row['target']) + "\t" + row['text'] + "\n")

with open(all_data_path, 'wb') as f:
    f.seek(0)
    f.truncate()

with open(all_data_path, 'a', encoding='utf-8') as f:
    for data in all_list:
        f.write(data)


def create_data_list(data_list_path):
    # 在生成数据之前，首先将eval_list.txt和train_list.txt清空
    with open(os.path.join(data_list_path, 'test_list.txt'), 'w', encoding='utf-8') as f_eval:
        f_eval.seek(0)
        f_eval.truncate()

    with open(os.path.join(data_list_path, 'train_list.txt'), 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate()

    with open(os.path.join(data_list_path, 'val_list.txt'), 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate()

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    with open(os.path.join(data_list_path, 'test_list.txt'), 'a', encoding='utf-8') as f_eval, open(
            os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train, open(
        os.path.join(data_list_path, 'val_list.txt'), 'a', encoding='utf-8') as f_val:
        for line in lines:
            if i % 10 == 0:
                f_eval.write(line)
            elif i % 10 == 1:
                f_val.write(line)
            else:
                f_train.write(line)
            i += 1

    print("数据列表生成完成！")

create_data_list(data_list_path)