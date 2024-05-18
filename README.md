## Highlights
  + Code Description: This project is based on the WeNet project, with the addition of the following files.
    > wenet/bin/train_dys.py, wenet/bin/infer_dys.py, wenet/transformer/resnet_3d.py, wenet/transformer/dys_model.py, wenet/dataset/dataset_dys.py, wenet/dataset/processor_dys.py, wenet/utils/executor_dys.py, wenet/utils/init_model_dys.py

  + Data Description：
    > This project uses the MSDM dataset. The training set and test set are divided by individuals, ensuring that data from the same person does not appear in the training and test sets simultaneously. Due to the diversity of dysarthria and the limitations in data size, it is recommended to average the models before inferring on the test set.
    > The data naming convention is as follows: N_M_10010_G4_task3_4_04. Here, N represents a healthy individual (S represents a patient), M represents male (F represents female), and 10010_G4_task3_4_04 is the identifier for this data entry.

## Task
  The challenge is a four-class classification task (normal, mild, moderate, and severe dysarthria) based on audio-visual data input, aimed at the automatic diagnosis and assessment of dysarthria severity.  
  Please refer to examples/MSDM/run.sh to run the baseline system.

## Results
  The evaluation model for the resnet18_conformer4 model is as follows:  
  segment result: {'accuracy': 0.6492376294591484, 'precision': 0.7164171337161716, 'recall': 0.6492376294591484, 'f1': 0.6154456959839272}
  final_F1: 6.13