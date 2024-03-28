import os
import types
import commentjson

from box import Box
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

        #


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
