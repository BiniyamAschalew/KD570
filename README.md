# Leveraging synthetic data for zero-shot knowledge distillation
<I>from <b>Distilling the Knowledge in a Neural Network</b> by Hinton et al. (2015)</i>  

<b>Spring 2024 CS570 Artificial Intelligence and Machine Learning<br>Team 2 Project Repository</b>  

Biniyam Aschalew Tolera, 20210733  
Minhajur Rahman Chowdhury Mahim, 20210753  
Bryan Nathanael Wijaya*, 20244078  

## Basic Information
The main implementations are in `KD`.
1. Teacher softmax temperature experiments were performed with `KD/temperature.py`
2. Student learning rate experiments were performed with `KD/learning_rate.py`
3. Transfer dataset experiments were performed with `KD/transfer_dataset_[DATASET_NAME].py`
4. Synthetic dataset generations are done with the source codes in `KD/synthetic`

## Experiments
You will need to install PyTorch with the appropriate CUDA driver version and a couple more dependencies to run our experiments, which can be installed easily with `pip` or `conda`. We highly recommend you create a separate virtual environment for this repository using commands like the following before installing the dependencies.
```
conda create -n KD570
conda activate KD570
```

To run the experiments (1)-(3), go to the `KD` directory and run the relevant source code with the appropriate config file.
```
cd KD
python [PYTHON_SOURCE_CODE_NAME].py --config_dir configs/train_configs/[CONFIG_YAML_NAME].yaml
```
The training configs in `configs/train_configs` are named with the following format:
```
student-[TEACHER_ARCH]-[STUDENT_ARCH]-[TRAINING_DATASET]-[TRANSFER_DATASET]-[SOFTMAX_TEMPERATURE]-[STUDENT_LEARNING_RATE]-[STUDENT_TRAINING_EPOCHS].yaml
```
The program will attempt to find the pre-trained teacher model upon execution, and when it is not found, it will train the teacher based on the provided configs.

To run the synthetic dataset generation, go to the `KD/synthetic` directory and run the relevant source code as it is.
```
cd KD/synthetic
python [PYTHON_SOURCE_CODE_NAME].py
```

## Supplementary Materials
Our model weights, real and synthetic datasets, logs, and experiment results are made available at [this link](https://drive.google.com/drive/folders/1olJpDZGBdqGfRMRGX4YWHssmSQlni3MQ?usp=sharing). To run the experiments properly, download this LFS and move its contents to the `KD` directory (e.g., the `data` directory is in the same hierarchical position as the `configs` directory).

The slides for this project are made available at [this link](https://drive.google.com/file/d/1_7xqqR90UwgvPAeDePMi4GJpRhfSYnVZ/view?usp=sharing). 
Feel free to contact us at [bryannwijaya@kaist.ac.kr](mailto:bryannwijaya@kaist.ac.kr) for any issues or inquiries.  
Happy coding!
