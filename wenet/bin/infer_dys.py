# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset_dys import Dataset
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.config import override_config
from wenet.utils.init_model_dys import init_model

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_dir', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    print(args)
    return args


def compute_metrics(results, labels):
    predicted = results.cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(labels, predicted)
    precision = precision_score(labels, predicted, average='weighted')
    recall = recall_score(labels, predicted, average='weighted')
    f1 = f1_score(labels, predicted, average='weighted')
    confusion_matrix_result = confusion_matrix(labels, predicted)
    final_score = 0.3*accuracy + 0.3*precision + 0.4*recall
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}, confusion_matrix_result, final_score

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 200
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           test_conf,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_model(configs)

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()

    f = open(os.path.join(args.result_dir, 'results.txt'), 'w')

    result_list, label_list = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, videos, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            videos = videos.to(device)
            target = target.to(device)
            target = target.to(torch.int64).squeeze()
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            results = model.decode(
                feats,
                videos,
                feats_lengths)
            hpys = torch.argmax(results, dim=1)
            if hpys.size(-1) != args.batch_size or target.size(-1) != args.batch_size:
                continue
            result_list.append(hpys)
            label_list.append(target)
            for i, key in enumerate(keys):
                f.write(f'{key}\t{hpys[i]}\n')
    results = torch.cat(result_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    aprf, confusion_matrix_result, final_score = compute_metrics(results, labels)
    f.write(f'{aprf}\n')
    f.close()
    print(aprf)
    print(confusion_matrix_result)
    print(final_score)


if __name__ == '__main__':
    main()