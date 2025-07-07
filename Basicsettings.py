from datetime import datetime
import os
import math
import numpy as np

# 获取当前日期，格式为 "Jan21"
current_date = datetime.now().strftime("%b%d")  # e.g., "Jan21"

# 基础参数
settings = {
    'minw': 2,
    'maxw': 500,
    'minp1': 1e-07,
    'minLinks': 2,
    'maxBlockLen': 10000,
    'minBlockDis': 20000,
    'width': 5000,
    'minFilereads': 800,
    'minBlockLen': 500,
    'CACBminLen':2000,
    'mergeDis': 2,
    'selfthreshold': 10000,
    'cutLinks': 20000,
    'a1': 1, 'a234': 1,
    'b1': 1, 'b234': 1,
    'r': 2, 'pp': 0.001,
    'sigma': 150,
    'delFlag': False,
    'delWid': 3000,
    'expandFlag': True,
    'minreadLen': 200 if True else 0,
    'methodname': current_date,  
    'prefix': 'test',
    'ABFlag': False,
    'keeptempFlag': True,
    'chromname': ['chr21']
}

# 输入和输出路径设置
settings['filedir'] = './input/test_multi.bedpe'
settings['outdir'] = os.path.join('./output/sbm', settings['methodname'])

# 创建输出目录结构
os.makedirs(settings['outdir'], exist_ok=True)
os.makedirs(os.path.join(settings['outdir'], 'tempfiles'), exist_ok=True)

# 写入基础参数日志
with open(os.path.join(settings['outdir'], 'worklog.txt'), 'w') as f:
    f.write('---------Basic setting for model----------\n')
    for key, value in settings.items():
        f.write(f"{key}: {value}\n")
    f.write('\n')
    f.write('------------Data loading info------------\n')

# ----------------------------- MCMC Parameters -----------------------------
mcmcparas = {
    'groupNum': 10,
    'iters': 120,
    'subiter1': 1, #update_group_each
    'subiter2': 1, #update_L
    'subiter3': 1, #update_expand_shrink_list
    'subiter4': 2, #update_kth_block
    'subiter5': 1, #update_P_list
    'test_iters': 10,
    'test_subiter1': 0,
    'test_subiter2': 1,
    'test_subiter3': 3,
    'test_subiter4': 0,
    'test_subiter5': 0,
    'discard_iters': 20,
    'discard_subiter1': 0,
    'discard_subiter2': 1,
    'discard_subiter3': 1,
    'discard_subiter4': 0,
    'discard_subiter5': 0,
    'rep': 3,
    'rep_in': 4,
}

# 自动计算 MCMC 参数
mcmcparas['subiters'] = sum([mcmcparas[f'subiter{i}'] for i in range(1, 6)])
mcmcparas['test_subiters'] = sum([mcmcparas[f'test_subiter{i}'] for i in range(1, 6)])
mcmcparas['discard_subiters'] = sum([mcmcparas[f'discard_subiter{i}'] for i in range(1, 6)])
mcmcparas['threadnum'] = mcmcparas['rep'] * mcmcparas['rep_in']
mcmcparas['jump_iters'] = int(mcmcparas['iters'] / 6)
mcmcparas['sitenum'] = math.ceil(mcmcparas['iters'] / mcmcparas['jump_iters'])
mcmcparas['stopsite'] = np.full((mcmcparas['groupNum'], mcmcparas['rep']), mcmcparas['iters'] + 1)

# 写入 MCMC 参数日志
with open(os.path.join(settings['outdir'], 'mcmcparas.txt'), 'w') as f:
    f.write('---------MCMC setting for model----------\n')
    for key, value in mcmcparas.items():
        f.write(f"{key}: {value}\n")
    f.write('\n')