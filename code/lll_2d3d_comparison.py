#!/usr/bin/env python3
"""
LLL 周期对比实验（纯随机策略，融合 2D + 3D 动作）
极值保护优化版：彻底修复底层聚合警告，防溢出计算。
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

# 提速迭代限制
MAX_2D_ITER = 20
MAX_3D_OUTER_ITER = 8

# 忽略底层警告
np.seterr(all='ignore')

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
    iter_count = 0
    while iter_count < MAX_2D_ITER:
        iter_count += 1
        if np.dot(u, u) > np.dot(v, v): u, v = v, u
        norm_u_sq = np.dot(u, u) + EPS
        mu = np.dot(v, u) / norm_u_sq
        if not np.isfinite(mu): break
        m = round(mu)
        if m == 0: break
        v -= m * u
    return u, v

def gaussian_reduce_3d(u, v, w):
    u, v, w = u.copy().astype(np.float64), v.copy().astype(np.float64), w.copy().astype(np.float64)
    outer_count = 0
    improved = True
    while improved and outer_count < MAX_3D_OUTER_ITER:
        improved = False
        outer_count += 1
        u_new, v_new = gaussian_reduce_2d(u, v)
        if np.linalg.norm(u_new - u) > 1e-6 or np.linalg.norm(v_new - v) > 1e-6: improved = True
        u, v = u_new, v_new
        u_new, w_new = gaussian_reduce_2d(u, w)
        if np.linalg.norm(u_new - u) > 1e-6 or np.linalg.norm(w_new - w) > 1e-6: improved = True
        u, w = u_new, w_new
        v_new, w_new = gaussian_reduce_2d(v, w)
        if np.linalg.norm(v_new - v) > 1e-6 or np.linalg.norm(w_new - w) > 1e-6: improved = True
        v, w = v_new, w_new

    vecs = [u, v, w]
    norms = [np.linalg.norm(x) for x in vecs]
    order = np.argsort(norms)
    return vecs[order[0]], vecs[order[1]], vecs[order[2]]

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
    norms = np.maximum(norms, EPS)
    shortest = float(np.min(norms))
    log_def = math.log(gh / max(shortest, EPS))

    log_norm_sum = np.sum(np.log(norms))
    log_od = log_norm_sum - logdet
    if not np.isfinite(log_od) or log_od > 700:
        ortho_def = np.inf
    else:
        ortho_def = np.exp(log_od)

    max_cos = 0.0
    total_cos = 0.0
    count = 0
    n = basis.shape[0]
    for x in range(n):
        for y in range(x + 1, n):
            norm_prod = norms[x] * norms[y]
            if not np.isfinite(norm_prod) or norm_prod < EPS:
                cos_val = 0.0
            else:
                cos_val = np.dot(basis[x], basis[y]) / norm_prod
                cos_val = np.clip(cos_val, -1.0, 1.0)
            max_cos = max(max_cos, abs(cos_val))
            total_cos += abs(cos_val)
            count += 1
    mean_cos = total_cos / max(count, 1)
    return shortest, log_def, max_cos, mean_cos, ortho_def

def random_unimodular(n, max_val=5):
    L = np.eye(n, dtype=object)
    for i in range(1, n):
        for j in range(i): L[i, j] = np.random.randint(-max_val, max_val + 1)
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

    pair_actions = list(combinations(range(n), 2))
    triple_actions = list(combinations(range(n), 3))
    all_actions = pair_actions + triple_actions
    action_dim = len(all_actions)

    for step in range(1, steps + 1):
        idx = random.randint(0, action_dim - 1)
        action = all_actions[idx]
        if len(action) == 2:
            i, j = action
            v1, v2 = gaussian_reduce_2d(basis[i], basis[j])
            basis[i], basis[j] = v1, v2
        else:
            i, j, k = action
            v1, v2, v3 = gaussian_reduce_3d(basis[i], basis[j], basis[k])
            basis[i], basis[j], basis[k] = v1, v2, v3

        if lll_interval is not None and lll_interval > 0 and step % lll_interval == 0:
            basis, _ = lll_reduction(basis)

    shortest, log_def, max_cos, mean_cos, ortho_def = compute_metrics(basis, gh, logdet)
    return shortest, log_def, max_cos, mean_cos, ortho_def

# ===================== 多进程单次运行封装 =====================
def run_interval_task(interval, all_lattices, trials, steps):
    total_shortest, total_log_def, total_max_cos, total_mean_cos, total_ortho_def = [], [], [], [], []
    # 新增列表用于收集初始 LLL 长度
    total_lll_min = []

    for lat_info in all_lattices:
        # 收集当前格子的初始 LLL 最短向量（未做任何高斯约化）
        total_lll_min.append(lat_info['lll_min'])

        trial_shortest, trial_log_def, trial_max_cos, trial_mean_cos, trial_ortho_def = [], [], [], [], []
        for _ in range(trials):
            s, l, mx, mn, od = test_single_lattice(lat_info, interval, steps=steps)
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

    def safe_stats(arr):
        safe_arr = np.where(np.isinf(arr) | np.isnan(arr), np.nan, arr)
        return np.nanmean(safe_arr), np.nanstd(safe_arr)

    avg_shortest, std_shortest = safe_stats(total_shortest)
    avg_log_def, std_log_def = safe_stats(total_log_def)
    avg_max_cos, std_max_cos = safe_stats(total_max_cos)
    avg_mean_cos, std_mean_cos = safe_stats(total_mean_cos)
    avg_ortho_def, std_ortho_def = safe_stats(total_ortho_def)
    avg_lll_min, std_lll_min = safe_stats(total_lll_min)  # 计算初始 LLL 基线

    return {
        'lll_interval': str(interval),
        'avg_lll_min': avg_lll_min,
        'avg_shortest': avg_shortest,
        'std_shortest': std_shortest,
        'avg_log_def': avg_log_def,
        'std_log_def': std_log_def,
        'avg_max_cos': avg_max_cos,
        'std_max_cos': std_max_cos,
        'avg_mean_cos': avg_mean_cos,
        'std_mean_cos': std_mean_cos,
        'avg_ortho_def': avg_ortho_def,
        'std_ortho_def': std_ortho_def,
    }

# ===================== 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervals", type=str, default="0,1,2,5,10,20,50,100")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-lattices", type=int, default=50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="lll_2d3d_period_comparison_report.csv")
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

    print(f"\n开始 (2D+3D) 对比实验！测试格子数: {len(all_lattices)}, 测试循环(Trials): {args.trials}次")
    print(f"测试的 LLL 周期列表: {intervals}")
    print(f"【注意】表格中的 Avg LLL Min 为不做任何高斯约化时的初始 LLL 基线。\n")

    with Pool(processes=min(cpu_count(), len(intervals))) as pool:
        tasks = [(interval, all_lattices, args.trials, args.steps) for interval in intervals]
        future_results = [pool.apply_async(run_interval_task, task) for task in tasks]
        results_data = []
        for idx, future in enumerate(future_results):
            try:
                res = future.get()
                results_data.append(res)
                print(f"--- LLL周期: {res['lll_interval']} 测试完成！ LogDef = {res['avg_log_def']:.4f} (±{res['std_log_def']:.4f}) ---")
            except Exception as e:
                print(f"计算某个周期时发生异常: {e}")

    results_data.sort(key=lambda x: 0 if x['lll_interval'] == 'None' else int(x['lll_interval']))

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
        od_str = f"{row['avg_ortho_def']:.2e}" if np.isfinite(row['avg_ortho_def']) else "Inf"
        print(f"{row['lll_interval']:<10} | {row['avg_lll_min']:<15.6f} | {row['avg_shortest']:<15.6f} | {row['avg_log_def']:<15.4f} | {od_str:<15}")
    print("=" * 85)

if __name__ == "__main__":
    main()