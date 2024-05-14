### 代码说明
  本项目基于wenet工程，添加了以下文件
  wenet/bin/train_dys.py
  wenet/bin/infer_dys.py
  wenet/transformer/resnet_3d.py
  wenet/transformer/dys_model.py
  wenet/dataset/dataset_dys.py
  wenet/dataset/processor_dys.py
  wenet/utils/executor_dys.py
  wenet/utils/init_model_dys.py

### 数据说明
  本项目使用MSDM数据集，训练集、测试集和开发集是按照人来划分，因此同一个人的数据不会同时出现在训练集、测试集和开发集中。由于构音障碍的多样性以及数据量大小的限制，因此建议对模型进行平均后再对测试集进行推理。

### 任务
  此次挑战是一个输入为音视频数据的四分类任务（正常、轻度、中度、和重度构音障碍），用于自动诊断和评估构音障碍的严重程度。


### 结果
  resnet10_conformer3
  {'accuracy': 0.4977704257767549, 'precision': 0.6267377505204481, 'recall': 0.4977704257767549, 'f1': 0.4304755704283422}
