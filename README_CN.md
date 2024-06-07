## 强调
  + 环境：本项目基于wenet工程，所以需要参照wenet安装环境，此外需要安装torchvision
  + 代码说明：本项目基于WeNet工程，添加了以下文件(如果不想安装wenet可以将以下文件放入自己的wennet中)
    > wenet/bin/train_dys.py, wenet/bin/infer_dys.py, wenet/transformer/resnet_3d.py, wenet/transformer/dys_model.py, wenet/dataset/dataset_dys.py, wenet/dataset/processor_dys.py, wenet/utils/executor_dys.py, wenet/utils/init_model_dys.py

  + 数据说明：
    > 本项目使用MSDM数据集，训练集和测试集是按照人来划分，因此同一个人的数据不会同时出现在训练集、测试集中。由于构音障碍的多样性以及数据量大小的限制，因此建议对模型进行平均后再对测试集进行推理。
    > 数据命名规则，例如：N_M_10010_G4_task3_4_04，N代表正常人(S代表病人)，M代表男性(F代表女性)，10010_G4_task3_4_04是此条数据的编号。

## 任务
  此次挑战是一个输入为音视频数据的四分类任务（正常、轻度、中度、和重度构音障碍），用于自动诊断和评估构音障碍的严重程度。
  请参照examples/MSDM/run.sh运行baseline系统

## 结果
  resnet18_conformer4模型的评估模型如下：
  segment result: {'accuracy': 0.6492376294591484, 'precision': 0.7164171337161716, 'recall': 0.6492376294591484, 'f1': 0.6154456959839272}
  final_F1: 6.13
