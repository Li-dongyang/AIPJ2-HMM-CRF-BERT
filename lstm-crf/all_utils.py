from typing import NamedTuple
import yaml

class Config(NamedTuple):
    language: str
    model_name: str
    batch_size: int
    lr: float
    num_layers: int
    hidden_size: int
    dropout: float
    epochs: int
    device: int
    seed: int
    data_dir: str
    ckpt_dir: str
    train_file: str
    dev_file: str
    test_file: str
    output_file: str

    tokenizer: str
    max_length: int
    num_labels: int
    label2id: dict
    id2label: dict

    def get_config(path: str):
        with open(path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return Config(**config_dict)
    
