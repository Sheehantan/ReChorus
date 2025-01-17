import subprocess
import itertools
import os
import sys

# 定义超参数的选择范围
param_grid = {
    'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],  # Coefficient α for inductive bias and self-attention balance
    'c': [1, 3, 5, 7, 9],  # Frequency separation parameter
    'num_heads': [1, 2, 4],  # Number of attention heads
    'lr': [5e-4],  # Learning rate values
    'l2': [1e-3],  # L2 regularization
    'num_layers': [2,3],
}

# 所有超参数组合
param_combinations = list(itertools.product(*param_grid.values()))

# 你的其他固定参数
fixed_params = {
    'model_name': 'BSARec',
    'emb_size': 64,
    'history_max': 50,
    'train': 1,
    'num_workers': 0,
    'dataset': 'MovieLens_1M',
    'gpu': 0,
}

# 结果存储文件夹
results_dir = 'training_results'
os.makedirs(results_dir, exist_ok=True)

# 遍历所有超参数组合，生成并执行命令
for i, param_comb in enumerate(param_combinations):
    # 将参数名和对应值配对
    params = dict(zip(param_grid.keys(), param_comb))

    # 构建完整的命令行参数
    command = f"python main.py "
    for param, value in {**fixed_params, **params}.items():
        command += f"--{param} {value} "

    # 打印当前的命令
    print(f"Running command: {command}")
    
    # 设置输出日志文件
    log_file = os.path.join(results_dir, f"run_{i+1}.log")

    try:
        # 执行命令并捕获标准输出和标准错误
        with open(log_file, 'w') as f:
            print(f"Running command: {command}")
            # 使用subprocess.PIPE捕获输出
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # 获取标准输出和标准错误输出
            stdout, stderr = process.communicate()

            # 同时写入文件和终端
            if stdout:
                sys.stdout.write(stdout.decode('utf-8'))  # 打印到终端
                f.write(stdout.decode('utf-8'))  # 写入日志文件
            if stderr:
                sys.stderr.write(stderr.decode('utf-8'))  # 打印到终端
                f.write(stderr.decode('utf-8'))  # 写入日志文件

        print(f"Finished running with params {params}, output saved to {log_file}")
    except Exception as e:
        print(f"Error running command {command}: {e}")
