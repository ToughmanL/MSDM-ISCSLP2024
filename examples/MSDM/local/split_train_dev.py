#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 split_test_train.py
* @Time 	:	 2024/04/28
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 xk.liu@siat.ac.cn
* @License	:	 (C)Copyright 2022-2025, lxk&AISML
* @Desc   	:	 None
'''

import os
import json


class SplitTestTrain():
  def __init__(self) -> None:
    self.file_list = ['data.list']
    self.data_dict = {}
    # 本次提供的实验数据是使用以下人作为验证集
    self.dev_persons = ['N_M_10010', 'N_F_10024', 'S_M_00024', 'S_M_00043', 'S_F_00054', 'S_M_00009']

  def get_data_info(self, dir):
    for file in self.file_list:
      line_set = set()
      self.data_dict[file] = []
      if not os.path.exists(os.path.join(dir, file)):
        continue
      with open(os.path.join(dir, file), 'r') as f:
        for line in f:
          if line in line_set:
            continue
          else:
            line_set.add(line)
            self.data_dict[file].append(line)
  
  def is_test_person(self, line):
    name = json.loads(line)['key']
    name_ll = name.split('_')
    if 'repeat' in name:
      person = name_ll[1] + '_' + name_ll[3] + '_' + name_ll[2]
    else:
      person = name_ll[0] + '_' + name_ll[2] + '_' + name_ll[1]
    if person in self.dev_persons:
      return 'dev'
    else:
      return 'train'
      
  def write_data(self, data_set, file):
    with open(file, 'w') as f:
      for line in data_set:
        f.write(line)

  def split_data(self, train_dir, dev_dir):
    test_set, dev_set, train_set = set(), set(), set()
    for file in self.file_list:
      for line in self.data_dict[file]:
        ttd = self.is_test_person(line)
        if ttd == 'dev':
          dev_set.add(line)
        else:
          train_set.add(line)
    
    self.write_data(dev_set, os.path.join(dev_dir, 'data.list'))
    self.write_data(train_set, os.path.join(train_dir, 'data.list'))
    return train_set, test_set, dev_set
  

if __name__ == '__main__':
  dir = 'data/'
  train_dir = 'data/train'
  dev_dir = 'data/dev'
  if not os.path.exists(train_dir):
    os.makedirs(train_dir)
  if not os.path.exists(dev_dir):
    os.makedirs(dev_dir)
  std = SplitTestTrain()
  std.get_data_info(dir)
  train_set, dev_set = std.split_data(train_dir, dev_dir)

