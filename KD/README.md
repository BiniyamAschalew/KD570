# Large File Storage for Spring 2024 CS570 AI and ML Project 
```
Leveraging synthetic data for zero-shot knowledge distillation
Team 2: Biniyam Tolera Aschalew, Minhajur Rahman Chowdhury Mahim, Bryan Nathanael Wijaya
```

## ```data```

This directory contains all the datasets used in our project, which are CIFAR10, ImageNet32, MNIST, and SVHN.  
Additionally, synthetic images were also generated for these datasets, saved at the `synthetic/[DATASET]/[GENERATION-METHOD]` subdirectories as npz files.  
For each method, a total of 120,000 normalized synthetic images with the same dimensions as the original images in the dataset were generated.  
In particular, the dimensions for MNIST is (1, 28, 28), SVHN is (3, 32, 32), CIFAR10 is (3, 32, 32), and ImageNet32 is (3, 32, 32).  

## ```logs```
This directory contains all the logs of the experiments in our project. 
- The `ablation1` subdirectory contains the logs generated in Ablation Study 1 for Softmax Temperature 
- The `ablation2` subdirectory contains the logs generated in Ablation Study 2 for Student Learning Rate
- The `ablation3` subdirectory contains the logs generated in Ablation Study 3 for Transfer Dataset
- The `experiments` subdirectory contains the logs generated in Generalizability for Other Dataset
- The `synthetic` subdirectory contains the logs generated for generating the synthetic images in Generalizability for Other Datasets
- The `teachers` subdirectory contains the logs generated when training the teacher models

The file naming format is as follows:

- Teacher models in `teachers`
    ```
    teacher-[TEACHER_ARCH]-[TRAINING_DATASET]-[TEACHER_LEARNING_RATE]-[TEACHER_TRAINING_EPOCHS]/*.txt
    ```
- Student models in `ablation*` and `experiments`
    ```
    student-[TEACHER_ARCH]-[STUDENT_ARCH]-[TRAINING_DATASET]-[TRANSFER_DATASET]-[SOFTMAX_TEMPERATURE]-[STUDENT_LEARNING_RATE]-[STUDENT_TRAINING_EPOCHS]/*.txt
    ```
- Synthetic models in `synthetic`
    ```
    [TRAINING_DATASET(s)]-[MODEL_ARCH]-epoch-[TRAINING_EPOCHS]/*.txt
    ```

where:

- `TEACHER_ARCH`: ResNet101
- `STUDENT_ARCH`: ResNet18
- `MODEL_ARCH`: cGAN, pix2pix, CycleGAN, DDPM
- `TRAINING_DATASET`: MNIST, SVHM, CIFAR10, ImageNet32
- `TRANSFER_DATASET`: anything in `TRAINING_DATASET`, None, (+)Noise, (+)ActMax, (+)cGAN, (+)pix2pix, (+)CycleGAN, (+)DDPM
- `SOFTMAX_TEMPERATURE`, `*LEARNING_RATE`, `*TRAINING_EPOCHS`: any number

## ```trained_models```
This directory contains all the models trained and used in our project. 
- The `ablation1` subdirectory contains the student models generated in Ablation Study 1 for Softmax Temperature 
- The `ablation2` subdirectory contains the student models generated in Ablation Study 2 for Student Learning Rate
- The `ablation3` subdirectory contains the student models generated in Ablation Study 3 for Transfer Dataset
- The `experiments` subdirectory contains the student models generated in Generalizability for Other Dataset
- The `synthetic` subdirectory contains the synthetic models generated or used for generating the synthetic images in Generalizability for Other Datasets
- The `teachers` subdirectory contains the teacher models generated throughout the experiments

The file naming format is as follows:

- Teacher models in `teachers`
    ```
    teacher-[TEACHER_ARCH]-[TRAINING_DATASET]-[TEACHER_LEARNING_RATE]-[TEACHER_TRAINING_EPOCHS].pth
    teacher-[TEACHER_ARCH]-[TRAINING_DATASET]-pretrained.pth
    ```
- Student models in `ablation*` and `experiments`
    ```
    student-[TEACHER_ARCH]-[STUDENT_ARCH]-[TRAINING_DATASET]-[TRANSFER_DATASET]-[SOFTMAX_TEMPERATURE]-[STUDENT_LEARNING_RATE]-[STUDENT_TRAINING_EPOCHS].pth
    ```
- Synthetic models in `synthetic`
    ```
    [TRAINING_DATASET(s)]-[MODEL_ARCH]-epoch-[TRAINING_EPOCHS].pth
    ```

where `MODEL_ARCH`: cGAN-{D,G}, pix2pix-{D,G}, CycleGAN-{Dx,Dy,F,G}, DDPM, while all other variables are the same.


## ```configs``` in the main GitHub repository
This directory contains the configurations used in our project.
- The `dataset_configs` contains the configuration of the datasets
- The `model_configs` contains the configuration of the models
- The `train_configs` contains the configuration of the training of teachers and students
    - This subdirectory is further organized into subdirectories and named like the `logs` directory, except the `/*.txt` in the filenames are changed to `.yaml`