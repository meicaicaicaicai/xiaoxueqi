# 测试能不能跑

import subprocess
import sys

modules = ['data_analysis', 'model', 'evaluate']
for m in modules:
    print(f'=== Testing {m}.py ===')
    rc = subprocess.call([sys.executable, f'scripts/{m}.py'])
    if rc != 0:
        print(f'{m}.py failed with exit code {rc}')
        break
    print(f'{m}.py passed\n')