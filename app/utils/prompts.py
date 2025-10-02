from functools import lru_cache
from .config import config
import os
import yaml


@lru_cache()
def get_prompt_template(prompt_name: str):
    with open(config["prompt_templates"]["path"] + prompt_name + ".yaml", "r", encoding="utf-8") as f:
        prompt = yaml.safe_load(f)
    return prompt
