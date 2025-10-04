import json
from types import SimpleNamespace

from functools import lru_cache


@lru_cache()
def get_conf():
    with open('./config.json', 'r') as f:
        json_data = f.read()
        config_object = json.loads(
            json_data, 
            object_hook=lambda d: SimpleNamespace(**d)
        )
        
    return config_object

conf = get_conf()
