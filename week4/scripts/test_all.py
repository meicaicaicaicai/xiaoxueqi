# 测试能不能跑

# test_all.py
import subprocess
import sys
import os

# 切到 scripts 目录
os.chdir(os.path.dirname(__file__))

modules = ['data_analysis', 'model', 'evaluate']
for m in modules:
    print(f'=== Testing {m}.py ===')
    rc = subprocess.call([sys.executable, f'{m}.py'])
    if rc != 0:
        print(f'{m}.py failed with exit code {rc}')
        break
    print(f'{m}.py passed\n')