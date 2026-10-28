"""
50维 三个模型对比验证脚本

模型:
  1. V50-CONTIG.2 (K=35, MLP)  -> best_ppo_50d_contig_v2.pth
  2. V50-CONTIG.3 (K=40, MLP)  -> best_ppo_50d_contig_v3_k40.pth
  3. V50-ST       (K=35, Attn) -> best_ppo_50d_st.pth

用法:
  python evaluate_50d_compare_all.py --max-lattices 20 --samples 2
  python evaluate_50d_compare_all.py --max-lattices 100 --samples 4 --with-bkz45
"""
import os, glob, math, random, time, argparse, pickle
import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL, BKZ, SVP
except ImportError:
    raise ImportError("请安装 fpylll")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASIS_NUM_50 = 50
LATTICE_DIR = r"../demo" 
FILE_PATTERN = "random_lattice_50_*"

START_LLL_DELTA = 0.7
LLL_FALLBACK_INTERVAL = 100
RESET_UNIMODULAR_MAX_VAL = 3

# ===================== 网络结构定义 =====================
# 1. MLP 结构 (用于 v2, v3)
STATE_DIM_FLAT = BASIS_NUM_50 + BASIS_NUM_50 * (BASIS_NUM_50 - 1) // 2 + 2  # 1277
HIDDEN_DIM = 640


class ActorCritic50_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM_FLAT, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.selector = nn.Linear(HIDDEN_DIM, BASIS_NUM_50)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0), -10.0, 10.0)
        h = self.fc2(self.fc1(x))
        return torch.nan_to_num(self.selector(h), nan=0.0), torch.nan_to_num(self.critic(h).squeeze(-1), nan=0.0)


# 2. Set Transformer 结构 (用于 ST)
ROW_FEAT_DIM = 100
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2


class ActorCritic50_ST(nn.Module):
    def __init__(self):
        super().__init__()
        self.row_proj = nn.Sequential(nn.Linear(ROW_FEAT_DIM, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM * 2,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.selector = nn.Linear(EMBED_DIM, 1)
        self.critic = nn.Sequential(nn.Linear(EMBED_DIM + 2, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        B = x.shape[0]
        row_feat = x[:, :BASIS_NUM_50 * ROW_FEAT_DIM].view(B, BASIS_NUM_50, ROW_FEAT_DIM)
        global_feat = x[:, -2:]
        h = self.row_proj(row_feat)
        h = self.transformer(h)
        rows_logits = self.selector(h).squeeze(-1)
        h_pool = h.mean(dim=1)
        value = self.critic(torch.cat([h_pool, global_feat], dim=-1)).squeeze(-1)
        return torch.nan_to_num(rows_logits, nan=0.0), torch.nan_to_num(value, nan=0.0)


# ===================== 工具函数 =====================
def norm_sq(row): return sum(int(x) * int(x) for x in row)


def min_norm_sq(basis): return min(norm_sq(row) for row in basis)


def has_zero_row(basis): return any(all(int(x) == 0 for x in row) for row in basis)


def sqrt_float_from_sq(nsq): return 0.0 if nsq <= 0 else math.exp(0.5 * math.log(nsq))


def load_lattice_50d(file_name):
    fp = os.path.join(LATTICE_DIR, file_name)
    if not os.path.exists(fp):
        for ext in ['.txt', '']:
            cand = os.path.join(LATTICE_DIR, file_name + ext)
            if os.path.exists(cand): fp = cand; break
        else:
            raise FileNotFoundError(f"未找到 {file_name}")
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


def lll_reduction_safe(basis, delta=0.99):
    n, m = len(basis), len(basis[0])
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    try:
        result = LLL.reduction(B_mat, delta=delta)
        B_out = result[0] if isinstance(result, tuple) else result
        new_basis = [[int(B_out[i, j]) for j in range(m)] for i in range(n)]
        return basis if has_zero_row(new_basis) else new_basis
    except Exception:
        return basis


def enumerate_sub_lattice(basis_50d, selected_rows):
    k = len(selected_rows)
    if k < 2: return None, -1
    try:
        sub_basis = [[int(x) for x in basis_50d[r]] for r in selected_rows]
        B = IntegerMatrix.from_matrix(sub_basis)
        par = BKZ.Param(block_size=k, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
        BKZ.reduction(B, par)
        best_sq, best_v = float('inf'), None
        for i in range(k):
            row = [int(B[i, j]) for j in range(BASIS_NUM_50)]
            sq = sum(x * x for x in row)
            if 0 < sq < best_sq: best_sq, best_v = sq, row
        return (best_v, best_sq) if best_v else (None, -1)
    except Exception:
        return None, -1


def compute_state_flat(basis, episode_start_sq, step, max_steps):
    n = BASIS_NUM_50
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]
    cos_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            dot = sum(int(basis[i][k]) * int(basis[j][k]) for k in range(n))
            if dot == 0: cos_vals.append(0.0); continue
            log_cos_sq = 2 * math.log(abs(dot)) - math.log(norm_sqs[i]) - math.log(norm_sqs[j])
            cos_vals.append(min(abs(math.exp(0.5 * min(0.0, log_cos_sq))), 1.0))
    cur_min = min_norm_sq(basis)
    start_ratio = math.log(max(episode_start_sq, 1)) - math.log(
        max(cur_min, 1)) if episode_start_sq > 0 and cur_min > 0 else 0.0
    state = np.concatenate([np.array(log_lengths, dtype=np.float32), np.array(cos_vals, dtype=np.float32),
                            np.array([start_ratio, step / max(max_steps, 1)], dtype=np.float32)])
    return np.clip(np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(np.float32)


def compute_state_matrix(basis, episode_start_sq, step, max_steps):
    n = BASIS_NUM_50
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]
    max_abs = max(abs(int(x)) for row in basis for x in row)
    scale = 1.0 / max_abs if max_abs > 0 else 1.0
    B_f = [[float(int(x)) * scale for x in row] for row in basis]
    mu = [[0.0] * n for _ in range(n)]
    b_star, norm_sq_bstar = [], []
    for i in range(n):
        bi = B_f[i][:]
        for j in range(i):
            norm_j = norm_sq_bstar[j]
            if norm_j < 1e-300: continue
            mu_val = sum(bi[k] * b_star[j][k] for k in range(n)) / norm_j
            mu[i][j] = max(min(mu_val, 5.0), -5.0)
            for k in range(n): bi[k] -= mu_val * b_star[j][k]
        b_star.append(bi)
        norm_sq_bstar.append(sum(x * x for x in bi))
    feat = np.zeros((n, ROW_FEAT_DIM), dtype=np.float32)
    for i in range(n):
        feat[i, 0] = log_lengths[i]
        idx = 1
        for j in range(n):
            if i == j: continue
            dot = sum(int(basis[i][k]) * int(basis[j][k]) for k in range(n))
            cos_val = 0.0
            if dot != 0:
                log_cos_sq = 2 * math.log(abs(dot)) - math.log(norm_sqs[i]) - math.log(norm_sqs[j])
                cos_val = min(abs(math.exp(0.5 * min(0.0, log_cos_sq))), 1.0)
            feat[i, idx] = cos_val
            feat[i, idx + 49] = mu[i][j] if i > j else mu[j][i]
            idx += 1
    cur_min = min_norm_sq(basis)
    start_ratio = math.log(max(episode_start_sq, 1)) - math.log(
        max(cur_min, 1)) if episode_start_sq > 0 and cur_min > 0 else 0.0
    global_feat = np.array([start_ratio, step / max(max_steps, 1)], dtype=np.float32)
    return np.clip(np.nan_to_num(feat, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0), global_feat


# ===================== 模型加载与评估 =====================
MODEL_CONFIGS = [
    {'name': 'V50-CONTIG.2 (K=35)', 'path': 'best_ppo_50d_contig_v2.pth', 'type': 'mlp', 'k': 35, 'steps': 200},
    {'name': 'V50-CONTIG.3 (K=40)', 'path': 'best_ppo_50d_contig_v3_k40.pth', 'type': 'mlp', 'k': 40, 'steps': 250},
    {'name': 'V50-ST (Attn, K=35)', 'path': 'best_ppo_50d_st.pth', 'type': 'st', 'k': 35, 'steps': 200}
]


def eval_single_model(model, model_type, info, k_select, max_steps, samples, seed):
    best_sq, best_start, zero_inc = None, None, 0
    with torch.no_grad():
        for s in range(samples):
            rng = random.Random(seed + s)
            basis = lll_reduction_safe(info['basis'], delta=START_LLL_DELTA)
            n = BASIS_NUM_50
            U = [[0] * n for _ in range(n)]
            for i in range(n): U[i][i] = 1
            for i in range(1, n):
                for j in range(i): U[i][j] = rng.randint(-RESET_UNIMODULAR_MAX_VAL, RESET_UNIMODULAR_MAX_VAL + 1)
            new_basis = [[sum(int(U[i][k]) * int(basis[k][j]) for k in range(n)) for j in range(n)] for i in range(n)]
            if not has_zero_row(new_basis): basis = new_basis

            ep_start_sq = min_norm_sq(basis)
            best_ep_sq = ep_start_sq

            for step in range(max_steps):
                if model_type == 'mlp':
                    state = compute_state_flat(basis, ep_start_sq, step, max_steps)
                    st_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                else:
                    feat, g_feat = compute_state_matrix(basis, ep_start_sq, step, max_steps)
                    state = np.concatenate([feat.flatten(), g_feat])
                    st_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

                logits, _ = model(st_tensor)
                start_idx = Categorical(logits=logits).sample().item()
                sel = [(start_idx + k) % n for k in range(k_select)]

                prev_b = [r[:] for r in basis]
                v, v_sq = enumerate_sub_lattice(basis, sel)
                if v is not None and v_sq > 0 and v_sq < best_ep_sq:
                    basis[np.argmax([norm_sq(r) for r in basis])] = v
                    if has_zero_row(basis): basis = prev_b; zero_inc += 1

                if LLL_FALLBACK_INTERVAL > 0 and (step + 1) % LLL_FALLBACK_INTERVAL == 0:
                    new_b = lll_reduction_safe(basis, delta=0.99)
                    if not has_zero_row(new_b): basis = new_b

                if has_zero_row(basis): basis = prev_b; zero_inc += 1

                new_sq = min_norm_sq(basis)
                if new_sq < best_ep_sq: best_ep_sq = new_sq

            if best_sq is None or best_ep_sq < best_sq:
                best_sq, best_start = best_ep_sq, ep_start_sq
    return {'final_sq': best_sq, 'start_sq': best_start, 'zero': zero_inc}


def build_infos(max_lattices, with_bkz45):
    files = sorted(glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN)))
    if not files: raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}")
    if max_lattices > 0: files = files[:max_lattices]
    print(f"从 {LATTICE_DIR} 加载 {len(files)} 个格文件...")
    infos = []
    for i, fp in enumerate(files):
        try:
            basis = load_lattice_50d(os.path.basename(fp))
            lll_min_sq = min_norm_sq(lll_reduction_safe(basis, delta=0.99))
            bkz_min_sq = lll_min_sq
            if with_bkz45:
                try:
                    B_mat = IntegerMatrix.from_matrix([[int(x) for x in r] for r in basis])
                    par = BKZ.Param(block_size=45, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
                    BKZ.reduction(B_mat, par)
                    bkz_min_sq = min_norm_sq([[int(B_mat[i, j]) for j in range(50)] for i in range(50)])
                except:
                    pass
            infos.append({'file': os.path.basename(fp), 'basis': basis, 'lll_min_sq': int(lll_min_sq),
                          'bkz30_min_sq': int(bkz_min_sq), 'svp_min_sq': int(bkz_min_sq)})
            if (i + 1) % 10 == 0: print(
                f"  [{i + 1}/{len(files)}] {os.path.basename(fp)}: LLL={sqrt_float_from_sq(lll_min_sq):.1f}, BKZ45={sqrt_float_from_sq(bkz_min_sq):.1f}")
        except Exception as e:
            print(f"  跳过 {fp}: {e}")
    return infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--with-bkz45", action="store_true", help="用 BKZ(45) 精化真值")
    args = parser.parse_args()

    # 1. 加载模型
    models = []
    for cfg in MODEL_CONFIGS:
        if not os.path.exists(cfg['path']):
            print(f"⚠️ 跳过 {cfg['name']} (未找到 {cfg['path']})")
            continue
        model = ActorCritic50_ST() if cfg['type'] == 'st' else ActorCritic50_MLP()
        model.load_state_dict(torch.load(cfg['path'], map_location=DEVICE))
        model.to(DEVICE).eval()
        models.append({'cfg': cfg, 'model': model})
        print(f"✓ 加载 {cfg['name']}: {cfg['path']}")

    if not models: raise RuntimeError("没有可用模型！")

    # 2. 加载数据
    infos = build_infos(args.max_lattices, args.with_bkz45)
    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)

    # 3. 评估
    all_results = {m['cfg']['name']: [] for m in models}
    total_zero = {m['cfg']['name']: 0 for m in models}

    print(f"\n开始评估 {len(infos)} 个格...\n")
    print("=" * 130)
    header = f"{'#':>3} | {'LLL':>8} | {'BKZ45':>8} |"
    for m in models: header += f" {m['cfg']['name'] + '_len':>15} | {m['cfg']['name'] + '_vsLLL':>12} |"
    print(header)
    print("=" * 130)

    t0 = time.time()
    for idx, info in enumerate(infos):
        lll = sqrt_float_from_sq(info['lll_min_sq'])
        bkz = sqrt_float_from_sq(info['svp_min_sq'])
        line = f"{idx + 1:>3} | {lll:>8.2f} | {bkz:>8.2f} |"

        for m in models:
            cfg = m['cfg']
            r = eval_single_model(m['model'], cfg['type'], info, cfg['k'], cfg['steps'], args.samples, args.seed)
            r['lll_sq'] = info['lll_min_sq']
            r['svp_sq'] = info['svp_min_sq']
            all_results[cfg['name']].append(r)
            total_zero[cfg['name']] += r['zero']

            final_len = sqrt_float_from_sq(r['final_sq'])
            vs_lll = (final_len - lll) / lll * 100 if lll > 0 else 0
            line += f" {final_len:>15.2f} | {vs_lll:>11.2f}% |"

        print(line)
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ... 进度 {idx + 1}/{len(infos)}, 耗时 {elapsed:.1f}s")

    print("=" * 130)
    print("\n===== 汇总对比 =====")
    print(f"{'指标':<20} |", end="")
    for m in models: print(f" {m['cfg']['name']:>20} |", end="")
    print("\n" + "-" * (23 + 23 * len(models)))

    def get_metrics(results):
        n = len(results)
        broke_lll = sum(1 for r in results if r['final_sq'] < r['lll_sq'])
        reached_bkz = sum(1 for r in results if r['final_sq'] <= r['svp_sq'])
        imp = [(sqrt_float_from_sq(r['start_sq']) - sqrt_float_from_sq(r['final_sq'])) / sqrt_float_from_sq(
            r['start_sq']) * 100 for r in results]
        gap_lll = [(sqrt_float_from_sq(r['final_sq']) - sqrt_float_from_sq(r['lll_sq'])) / sqrt_float_from_sq(
            r['lll_sq']) * 100 for r in results]
        gap_bkz = [(sqrt_float_from_sq(r['final_sq']) - sqrt_float_from_sq(r['svp_sq'])) / sqrt_float_from_sq(
            r['svp_sq']) * 100 for r in results]
        return broke_lll / n * 100, reached_bkz / n * 100, np.mean(imp), np.mean(gap_lll), np.mean(gap_bkz), min(
            r['final_sq'] for r in results), max(r['final_sq'] for r in results)

    metrics = {m['cfg']['name']: get_metrics(all_results[m['cfg']['name']]) for m in models}
    rows = [("突破 LLL (%)", 0), ("达到 BKZ45 (%)", 1), ("平均改进率 (%)", 2), ("平均距 LLL (%)", 3),
            ("平均距 BKZ45 (%)", 4), ("最优 final_sq", 5), ("最差 final_sq", 6)]

    for label, idx in rows:
        print(f"{label:<20} |", end="")
        for m in models:
            val = metrics[m['cfg']['name']][idx]
            fmt = f"{val:>20.4f}" if idx >= 5 else f"{val:>19.2f}%"
            print(f" {fmt} |", end="")
        print()


if __name__ == "__main__":
    main()