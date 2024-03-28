import os
import torch
import commentjson
import torcheval

from torchvision.transforms import v2
from typing import Tuple, Dict, List, Any
from pprint import pprint as pp


def _get_attr_name(key, config: Dict[str, Any]) -> str:
    if isinstance(config.get(key), Dict):
        return _get_attr_name(key=f"{key}_{config[key]}", config=config[key])
    return key


class ConfigManager:
    def __init__(self, path: str):
        config = commentjson.loads(open(file=path, mode="r", encoding="UTF-8").read())
        self.__set_dynamic_field(config)

        # check paths existence
        self.__check_output_path()


    def __set_dynamic_field(self, config: Dict[str, Any], max_recursive_level: int = 0) -> None:
        """
        Create class dynamic fields of configs with one recursive level
        """
        if max_recursive_level < 1:
            for key in config.keys():
                if isinstance(config.get(key), Dict):
                    self.__set_dynamic_field(
                        {f"{key}_{sub_key}": config[key][sub_key] for sub_key in config[key].keys() if config.get(key)},
                        max_recursive_level + 1
                    )
                else:
                    self.__set_dynamic_field({f"{key}": config[key]}, max_recursive_level + 1)
        else:
            # Create class field
            for k, v in config.items():
                setattr(self, k, v)
        return None

    def __check_output_path(self) -> None:
        """
        Check existence of checkpoint and log path
        If not exists, create dir as the following pattern:
            checkpoint/<MODEL_BASE>/<MODEL_NAME>
        """
        for path in ("checkpoint", "log"):
            # Add path to class attr
            k = f"{path.upper()}_PATH"
            v = os.path.join(os.getcwd(), "output", path, self.MODEL_BASE, self.MODEL_NAME)
            self.__dict__[k] = v

            # Create dir if not exists
            if not os.path.isdir(v):
                os.makedirs(v, 0o777, True)
                print(f"Dir for {k.lower()} {self.MODEL_NAME} is created.")

    def get_training_configs(self) -> Dict[str, Any]:
        # DATA in config file
        return {
            "DATASET_PATH": self.__dict__.pop("DATA_PATH"),
            "INPUT_SHAPE": self.__dict__.pop("DATA_INPUT_SHAPE", [224, 224, 3]),
            "TRAIN_SIZE": self.__dict__.pop("DATA_TRAIN_SIZE"),
            "NUM_WORKER": self.__dict__.pop("DATA_NUM_WORKER", 1),
            "TRANSFORM": self.__dict__.pop("DATA_TRANSFORM", None),
            "TARGET_TRANSFORM": self.__dict__.pop("DATA_TARGET_TRANSFORM", None),

            # CHECKPOINT in config file
            "checkpoint_path": self.__dict__.pop("CHECKPOINT_PATH"),
            "save_strategy": self.__dict__.pop("CHECKPOINT_SAVE_STRATEGY", "no"),
            "save_total_lim": self.__dict__.pop("CHECKPOINT_SAVE_TOTAL_LIM", 2),
            "load": self.__dict__.pop("CHECKPOINT_LOAD", False),
            "resume_name": self.__dict__.pop("CHECKPOINT_RESUME_NAME", None),
            "save_only_weight": self.__dict__.pop("CHECKPOINT_SAVE_ONLY_WEIGHT", True),
            "include_config": self.__dict__.pop("CHECKPOINT_INCLUDE_CONFIG", False),

            # MODEL in config file
            "model_base": self.__dict__.pop("MODEL_BASE"),
            "model_name": self.__dict__.pop("MODEL_NAME"),
            "model_pretrained": self.__dict__.pop("MODEL_PRETRAINED", False),
            "model_args": self.__dict__.pop("MODEL_ARGS"),

            # SOLVER in config file
            "optimizer": self.__dict__.pop("SOLVER_OPTIMIZER", None),
            "lr_scheduler": self.__dict__.pop("SOLVER_LR_SCHEDULER", None),
            "loss": self.__dict__.pop("SOLVER_LOSS", None),
            "metrics": self.__dict__.pop("SOLVER_METRICS", None),

            # EARLY_STOPPING in config file
            "EARLY_STOPPING_APPLY": self.__dict__.pop("EARLY_STOPPING_APPLY", False),
            "EARLY_STOPPING_ARGS": self.__dict__.pop("EARLY_STOPPING_ARGS", None),

            # LOGGING in config file
            "LOG_PATH": self.__dict__.pop("LOG_PATH"),
            "LOGGING_STRATEGY": self.__dict__.pop("LOGGING_STRATEGY", "epoch"),

            # MISC
            "SEED": self.__dict__.pop("SEED", 12345),
            "CUDA": self.__dict__.pop("CUDA", False),
            "BATCH_SIZE": self.__dict__.pop("BATCH_SIZE", 16),
            "TRAINING_EPOCHS": self.__dict__.pop("TRAINING_EPOCHS", 3),
            "EVALUATION_STRATEGY": self.__dict__.pop("EVALUATION_STRATEGY",  "no")
        }
