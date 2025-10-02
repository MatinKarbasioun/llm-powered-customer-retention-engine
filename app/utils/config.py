from functools import lru_cache
import json


@lru_cache()
def get_conf():
    """_summary_

    Returns:
        dict: the configuration file for the application
    """
    with open("../core/config.json", "r", encoding="utf-8") as f:
        conf = json.load(f)
    return conf


config: dict = get_conf()
