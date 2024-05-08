def write_config(config_dir, content):
    with open(config_dir, "w") as file:
        file.write(content)
    return

def create_model_config(
        
    model = "custom",
    pretrained = True,
    load_from_path = False,
    model_path = "./models",
    input_shape = [1, 28, 28],
    num_classes = 10,

    size = [16, 32],
    kernel_size = [3, 3],
    stride = [1, 1],
    padding = [0, 0],
    dropout = [0, 0],

    ):

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
                        dropout: {dropout}"""

    write_config(".configs/model_config/mnist_custom1.yaml", model_config)
    return model_config


def create_main_config(

    seed = 19,
    batch_size = 32,
    epochs = 2,
    learning_rate = 0.001,
    device = "cuda",
    multiprocess = True, 


    temperature = 1,
    distillation_weight = 0.5,
    ce_weight = 0.5,

    teacher_model_config = "model_configs/sample_model_config1.yaml",
    student_model_config = "model_configs/sample_model_config2.yaml",

    dataset = "MNIST",
    shape = [1, 28, 28],
    num_classes = 10,

    log_dir = "./logs",

    ):

    main_config = f"""seed: {seed}
                    batch_size: {batch_size}
                    epochs: {epochs}
                    learning_rate: {learning_rate}
                    device: {device}

                    temperature: {temperature}
                    distillation_weight: {distillation_weight}
                    ce_weight: {ce_weight}

                    teacher_model_config: {teacher_model_config}
                    student_model_config: {student_model_config}

                    dataset: {dataset}
                    shape: {shape}
                    num_classes: {num_classes}

                    log_dir: {log_dir}"""
    
    write_config("./main_config.yaml", main_config)
    return main_config