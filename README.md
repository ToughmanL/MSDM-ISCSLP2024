## Highlights
  + Code Description: This project is based on the WeNet project, with the addition of the following files.
    > wenet/bin/train_dys.py, wenet/bin/infer_dys.py, wenet/transformer/resnet_3d.py, wenet/transformer/dys_model.py, wenet/dataset/dataset_dys.py, wenet/dataset/processor_dys.py, wenet/utils/executor_dys.py, wenet/utils/init_model_dys.py

  + Data Description：
    > This project uses the MSDM dataset. The training set, test set, and development set are divided by individuals, ensuring that data from the same person does not appear in the training, test, and development sets simultaneously. Due to the diversity of dysarthria and the limitations in data size, it is recommended to average the models before inferring on the test set.

## Task

  The challenge is a four-class classification task (normal, mild, moderate, and severe dysarthria) based on audio-visual data input, aimed at the automatic diagnosis and assessment of dysarthria severity.  
  Please refer to examples/MSDM/run.sh to run the baseline system.

## Results
  The evaluation model for the resnet10_conformer3 model is as follows:  
  {'accuracy': 0.4977704257767549, 'precision': 0.6267377505204481, 'recall': 0.4977704257767549, 'f1': 0.4304755704283422}
