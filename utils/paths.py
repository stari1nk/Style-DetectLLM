import json
from typing import Literal, Optional
import os

from config import models_dir, SOURCE, CATEGORIES, MODELTYPE, dirs, NAME

def get_model_path(series:str, version:str = None,size: str = None, is_base: bool = False) -> str:
    model_para = {
        "series": series,
        "version": version,
        "size": size,
        "is_base": None if is_base else "chat",
    }
    model_name = '-'.join([str(v) for v in model_para.values() if v is not None])
    return models_dir[model_name]

def get_ImBD_dataset_path(category: str, modeltype: str, source:str) -> str:
    assert category in CATEGORIES and modeltype in MODELTYPE and source in SOURCE
    data_path = os.path.join(dirs["datasets"],"ImBD", category, modeltype, source, f'{source}_{category}_{modeltype}.raw_data.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    return data_path

def get_ImBD_train_path(num: int):
    assert num in [500,3000]
    return os.path.join(dirs["datasets"],"ImBD", f'ai_detection_{num}_polish.raw_data.json')

def get_data(name:str, train:bool, num: int = None,category: str = None, modeltype: str = None, source: str = None):
    assert name in NAME
    if name == "ImBD":
        if train:
            data_path = get_ImBD_train_path(num = num)
        else:
            data_path = get_ImBD_dataset_path(category,modeltype,source)
        datas = json.load(open(data_path,'r'))
        hum_seq, mac_seq = datas["original"], datas["rewritten"]
        labels = [0 for _ in range(len(hum_seq))] + [1 for _ in range(len(mac_seq))]
        seqs = hum_seq + mac_seq
        return seqs, labels
    else:
        return None,None
    
def get_activation_dir(name:str, train:bool, model_name:str, num: int = None,category: str = None, modeltype: str = None, source: str = None):
    assert name in NAME
    if name == "ImBD":
        if train:
            data_dir = os.path.join(dirs["activations"],"ImBD", str(num), model_name)
        else:
            data_dir = os.path.join(dirs["activations"],"ImBD", category, modeltype, source, model_name)
        return data_dir
    else:
        return None,None
