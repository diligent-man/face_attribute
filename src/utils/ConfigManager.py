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



