#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
* @File 	:	 model_migaration.py
* @Time 	:	 2022/11/17
* @Author	:	 lxk
* @Version	:	 1.0
* @Contact	:	 meiluosi@gmail.com
* @License	:	 (C)Copyright 2022-2025, lxk
* @Desc   	:	 None
'''

import logging
import os
import re
import yaml
import torch
from collections import OrderedDict
import datetime
from wenet.utils.init_model_dys import init_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.checkpoint import save_checkpoint

class ModelMigration():
  def __init__(self, yaml_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    with open(yaml_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    self.model = init_model(configs)

  def _load_dict(self, checkpoint_path, layer=None):
    checkpoint = {}
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if layer != None:
        new_checkpoint = {}
        for key, value in checkpoint.items():
            if layer in key:
                new_checkpoint[key] = value
        return new_checkpoint
    else:
        return checkpoint

  def _extract_params(self, source_dict, target_dict):
    migrat_dict = {}
    for k, v in source_dict.items():
      if k in target_dict.keys():
        if source_dict[k].shape == target_dict[k].shape:
          migrat_dict[k] = v
          # migrat_dict[k] = (v + target_dict[k])/2
        else:
          print("Inconsistent parameters {}".format(k))
    return migrat_dict
  
  def migration(self, transformer_path, resnet_path, targ_chek_path, new_chek_path):
    transformer_dict = self._load_dict(transformer_path, 'encoder')
    resnet_dict = self._load_dict(resnet_path, 'resnet3d')
    target_dict = self._load_dict(targ_chek_path)
    migrat_dict = self._extract_params(transformer_dict, target_dict)
    migrat_dict = self._extract_params(resnet_dict, target_dict)
    target_dict.update(target_dict)
    self.model.load_state_dict(migrat_dict, strict=False)
    save_checkpoint(self.model, new_chek_path)

if __name__ == "__main__":
  transformer_path = "exp/conformer_4drop4/avg20.pt"
  resnet_path = "exp/resnet_18/avg10.pt"
  targ_chek_path = "exp/conformer4_resnet18_migra/init.pt"
  new_chek_path = "exp/conformer4_resnet18_migra/migrate.pt"
  yaml_path = "exp/conformer4_resnet18_migra/train.yaml"
  MM = ModelMigration(yaml_path)
  MM.migration(transformer_path, resnet_path, targ_chek_path, new_chek_path)

