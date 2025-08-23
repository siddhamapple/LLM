from dataclasses import dataclass
from typing import Any, Dict
import yaml

@dataclass
class Config:
    raw: Dict[str, Any]

    def __getattr__(self, item):
        val = self.raw.get(item)
        if isinstance(val, dict):
            return Config(val)
        return val

def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(cfg)
