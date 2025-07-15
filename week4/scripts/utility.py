# 通用工具函数

import os
import json
import pandas as pd
from datetime import datetime

def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def save_json(obj, name):
    ensure_dir('output')
    with open(f'output/{name}.json', 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(df: pd.DataFrame, name):
    ensure_dir('output')
    df.to_csv(f'output/{name}.csv', index=False)

def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")