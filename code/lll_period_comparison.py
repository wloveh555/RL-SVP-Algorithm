#!/usr/bin/env python3
"""
LLL 周期对比实验（纯随机策略，无需 PPO）
核心修复：加入了二维高斯约简的 max_iter 死循环防锁机制。
修复2：采用稳定的 slogdet 预计算行列式，彻底修复正交缺陷输出 inf/nan 的问题。
修改：在最终报告和表格中加入初始 LLL 最短向量 (Avg LLL Min) 作为对比基线。
"""
import os
import sys
import glob
import math
import random
import argparse
import numpy as np
import csv
from itertools import combinations
from multiprocessing import Pool, cpu_count

# -------------------- 核心依赖 fpylll --------------------
try:
    from fpylll import IntegerMatrix, LLL
    FPLLL_AVAILABLE = True
except ImportError:
    raise ImportError("无法导入 fpylll，请确保在 WSL 环境中安装：pip install fpylll")

# -------------------- 基础配置 --------------------
DEVICE = "cpu"
EPS = 1e-8
GH = 1268.98
MAX_TEST_STEPS = 500
LATTICE_DIR = "../demo" 
FILE_PATTERN = "random_lattice_20_*"

# ===================== 数学工具函数 =====================
def load_lattice(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = [float(x) for x in content.split() if x]
    n = int(round(math.sqrt(len(nums))))
    return np.array(nums, dtype=np.float64).reshape(n, n)

def process_single_lattice(fp):
    try:
        orig = load_lattice(fp)
        sign, logdet = np.linalg.slogdet(orig)
        if sign == 0: return None, os.path.basename(fp), "行列式为0"
        n = orig.shape[0]
        mink_bound = math.sqrt(n) * math.exp(logdet / n)
        lll_basis, _ = lll_reduction(orig)
        lll_min = float(np.min(np.linalg.norm(lll_basis, axis=1)))
        info = {
            'basis': orig, 'lll_min': lll_min, 'mink_bound': max(mink_bound, EPS),
            'dim': n, 'filename': os.path.basename(fp), 'logdet': logdet
        }
        return info, os.path.basename(fp), None
    except Exception as e:
        return None, os.path.basename(fp), str(e)

def gaussian_reduce_2d(u, v):
    u, v = u.copy().astype(np.float64), v.copy().astype(np.float64)
    max_iter = 1000
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        if np.linalg.norm(u) > np.linalg.norm(v):
            u, v = v, u
        norm_u_sq = np.dot(u, u) + EPS
        mu = np.dot(v, u) / norm_u_sq
        if not np.isfinite(mu):
            break
        m = round(mu)
        if m == 0:
            break
        v -= m * u
    return u, v

def lll_reduction(basis, delta=0.99):
    n = basis.shape[0]
    int_basis_list = [[int(round(basis[i, j])) for j in range(n)] for i in range(n)]
    B_mat = IntegerMatrix.from_matrix(int_basis_list)
    result = LLL.reduction(B_mat, delta=delta)
    if isinstance(result, tuple) and len(result) == 2:
        B_out, T_out = result
    else:
        B_out = result
        T_out = IntegerMatrix.identity(n)
    B_final = np.array([list(row) for row in B_out], dtype=np.float64)
    T_final = np.array([list(row) for row in T_out], dtype=np.float64)
    if B_final.ndim == 1: B_final = B_final.reshape(n, n)
    if T_final.ndim == 1: T_final = T_final.reshape(n, n)
    return B_final, T_final

def compute_metrics(basis, gh, logdet):
    norms = np.linalg.norm(basis, axis=1)
    shortest = float(np.min(norms))

    log_def = math.log(gh / max(shortest, EPS))

    log_norm_sum = np.sum(np.log(norms))
    log_od = log_norm_sum - logdet

    if log_od > 700:
        ortho_def = np.inf
    else:
        ortho_def = np.exp(log_od)

    max_cos = 0.0
    total_cos = 0.0
    count = 0
    n = basis.shape[0]
    for x in range(n):
        for y in range(x + 1, n):
            cos = np.dot(basis[x], basis[y]) / (norms[x] * norms[y] + EPS)
            max_cos = max(max_cos, abs(cos))
            total_cos += abs(cos)
            count += 1
    mean_cos = total_cos / max(count, 1)

    return shortest, log_def, max_cos, mean_cos, ortho_def

def random_unimodular(n, max_val=5):
    L = np.eye(n, dtype=object)
    for i in range(1, n):
        for j in range(i):
            L[i, j] = np.random.randint(-max_val, max_val + 1)
    perm = np.random.permutation(n)
    P = np.zeros((n, n), dtype=np.int64)
    P[np.arange(n), perm] = 1
    U_obj = L @ P.astype(object)
    U = U_obj.astype(np.float64)
    try:
        det = np.linalg.det(U)
        return U if abs(round(det)) == 1 else np.eye(n, dtype=np.float64)
    except Exception:
        return np.eye(n, dtype=np.float64)

# ===================== 单格测试逻辑 =====================
def test_single_lattice(lattice_info, lll_interval, steps=MAX_TEST_STEPS):
    orig_basis = lattice_info['basis']
    n = lattice_info['dim']
    gh = GH
    logdet = lattice_info['logdet']

    U = random_unimodular(n)
    basis = orig_basis @ U
    for i in range(n):
        for j in range(i + 1, n):
            v1, v2 = gaussian_reduce_2d(basis[i], basis[j])
            basis[i], basis[j] = v1, v2
    basis = basis.astype(np.float64)

    action_list = list(combinations(range(n), 2))
    action_dim = len(action_list)

    for step in range(1, steps + 1):
        idx = random.randint(0, action_dim - 1)
        i, j = action_list[idx]
        v1, v2 = gaussian_reduce_2d(basis[i], basis[j])
        basis[i], basis[j] = v1, v2

        if lll_interval is not None and lll_interval > 0 and step % lll_interval == 0:
            basis, _ = lll_reduction(basis)

    shortest, log_def, max_cos, mean_cos, ortho_def = compute_metrics(basis, gh, logdet)
    return shortest, log_def, max_cos, mean_cos, ortho_def

# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=str, default="0,1,2,5,10,20,50,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-lattices", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="lll_period_comparison_report.csv")
    args = parser.parse_args()

    intervals = []
    for s in args.intervals.split(','):
        s = s.strip()
        if s == "0": intervals.append(None)
        else: intervals.append(int(s))

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN))
    if not files: raise FileNotFoundError(f"未找到格文件：{LATTICE_DIR}")

    print(f"找到 {len(files)} 个格数据，开始多线程预处理...")
    with Pool(min(cpu_count(), len(files))) as pool:
        results = pool.map(process_single_lattice, files[:args.max_lattices])

    all_lattices = []
    print("=" * 80)
    for info, fname, err in results:
        if err:
            print(f"跳过 {fname}: {err}")
        else:
            all_lattices.append(info)
            print(f"[Baseline] 文件: {fname:40s} | LLL_min: {info['lll_min']:.6f} | Minkowski: {info['mink_bound']:.2f}")
    print("=" * 80)
    if not all_lattices: raise RuntimeError("未加载到任何有效的格数据")

    print(f"\n开始对比实验！测试格子数: {len(all_lattices)}, 每次测试步数: {args.steps}, 测试循环(Trials): {args.trials}次")
    print(f"测试的 LLL 周期列表: {intervals}\n")

    # 收集整个数据集的初始 LLL 基线
    total_lll_min = [lat['lll_min'] for lat in all_lattices]

    results_data = []

    # 遍历每个 LLL 周期
    for interval in intervals:
        print(f"--- 正在测试 LLL 周期: {interval if interval is not None else '无LLL'} ---")

        total_shortest, total_log_def, total_max_cos, total_mean_cos, total_ortho_def = [], [], [], [], []

        # 遍历选中的格子
        for idx, lat_info in enumerate(all_lattices):
            sys.stdout.write(f"    正在处理格子 {idx + 1}/{len(all_lattices)} : {lat_info['filename']} ... \r")
            sys.stdout.flush()

            trial_shortest, trial_log_def, trial_max_cos, trial_mean_cos, trial_ortho_def = [], [], [], [], []
            for _ in range(args.trials):
                s, l, mx, mn, od = test_single_lattice(lat_info, interval, steps=args.steps)
                trial_shortest.append(s)
                trial_log_def.append(l)
                trial_max_cos.append(mx)
                trial_mean_cos.append(mn)
                trial_ortho_def.append(od)

            total_shortest.append(np.mean(trial_shortest))
            total_log_def.append(np.mean(trial_log_def))
            total_max_cos.append(np.mean(trial_max_cos))
            total_mean_cos.append(np.mean(trial_mean_cos))
            total_ortho_def.append(np.mean(trial_ortho_def))

        print(f"    当前周期格子处理完成！")

        # 计算平均 LLL 最短向量（每次周期都算，保持一致）
        avg_lll_min = np.mean(total_lll_min)

        results_data.append({
            'lll_interval': str(interval),
            'avg_lll_min': avg_lll_min,
            'avg_shortest': np.mean(total_shortest),
            'std_shortest': np.std(total_shortest),
            'avg_log_def': np.mean(total_log_def),
            'std_log_def': np.std(total_log_def),
            'avg_max_cos': np.mean(total_max_cos),
            'std_max_cos': np.std(total_max_cos),
            'avg_mean_cos': np.mean(total_mean_cos),
            'std_mean_cos': np.std(total_mean_cos),
            'avg_ortho_def': np.mean(total_ortho_def),
            'std_ortho_def': np.std(total_ortho_def),
        })

        print(f"  Interval {interval}: LogDef = {results_data[-1]['avg_log_def']:.4f} (±{results_data[-1]['std_log_def']:.4f})")

    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['lll_interval', 'avg_lll_min', 'avg_shortest', 'std_shortest',
                      'avg_log_def', 'std_log_def',
                      'avg_max_cos', 'std_max_cos',
                      'avg_mean_cos', 'std_mean_cos',
                      'avg_ortho_def', 'std_ortho_def']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_data:
            writer.writerow(row)

    print(f"\n🎉 对比实验完成！详细数据已保存至: {args.output}")

    print("\n===== 最终对比摘要 =====")
    print(f"{'LLL周期':<10} | {'Avg LLL Min':<15} | {'Avg Shortest':<15} | {'Avg LogDef':<15} | {'Avg OrthoDef':<15}")
    print("-" * 85)
    for row in results_data:
        od_str = f"{row['avg_ortho_def']:.2e}" if row['avg_ortho_def'] != np.inf else "Inf"
        print(f"{row['lll_interval']:<10} | {row['avg_lll_min']:<15.6f} | {row['avg_shortest']:<15.6f} | {row['avg_log_def']:<15.4f} | {od_str:<15}")
    print("=" * 85)

if __name__ == "__main__":
    main()