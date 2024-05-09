from typing import List

def write_config(config_dir: str, content: str):

    with open(config_dir, "w") as file:
        file.write(content)

def create_model_config(
        
    model: str = "custom",
    pretrained: bool = True,
    load_from_path: bool = False,
    model_path: str = "./models",
    input_shape: List[int] = [1, 28, 28],
    num_classes: int = 10,

    size: List[int] = [16, 32],
    kernel_size: List[int] = [3, 3],
    stride: List[int] = [1, 1],
    padding: List[int] = [0, 0],
    dropout: List[float] = [0, 0],

    save_path: str = "",
    config_file_name: str = ".configs/sample_model_config.yaml"

    ) -> str:

    assert(len(size) == len(kernel_size) == len(stride) == len(padding) == len(dropout))

    model_config =  f"""model: {model}
                        pretrained: {pretrained} 
                        load_from_path: {load_from_path}
                        model_path: {model_path}

                        input_shape: {input_shape}
                        num_classes: {num_classes}

                        size: {size}
                        kernel_size: {kernel_size}
                        stride: {stride}
                        padding: {padding}
                        dropout: {dropout}
                        save_path: {save_path}"""

    write_config(config_file_name, model_config)
    return model_config


def create_main_config(

    seed: int = 19,
    batch_size: int = 32,
    epochs: int = 2,
    learning_rate: float = 0.001,
    device: str = "cuda",
    multiprocess: bool = True,

    temperature: float = 1,
    distillation_weight: float = 0.5,
    ce_weight: float = 0.5,

    teacher_model_config: str = "model_configs/sample_model_config1.yaml",
    student_model_config: str = "model_configs/sample_model_config2.yaml",

    dataset: str = "MNIST",
    shape: List[int] = [1, 28, 28],
    num_classes: int = 10,

    log_dir: str = "./logs",
    config_file_name: str = ".configs/sample_config.yaml"

    ) -> str:

    main_config = f"""seed: {seed}
                    batch_size: {batch_size}
                    epochs: {epochs}
                    learning_rate: {learning_rate}
                    device: {device}
                    multiprocess: {multiprocess}

                    temperature: {temperature}
                    distillation_weight: {distillation_weight}
                    ce_weight: {ce_weight}

                    teacher_model_config: {teacher_model_config}
                    student_model_config: {student_model_config}

                    dataset: {dataset}
                    shape: {shape}
                    num_classes: {num_classes}

                    log_dir: {log_dir}"""
    
    write_config(config_file_name, main_config)
    return main_config