"""
40维 混合动作训练代码 V40-MIX.1（NaN 修复版）

修复内容（V40-MIX → V40-MIX.1）：
  ★ 修复 1：ActorCritic40Mix.forward 输出前加 nan_to_num
  ★ 修复 2：get_action 里 log_probs 加 nan_to_num
  ★ 修复 3：PPO update ratio 加 clamp(-10, 10)
  ★ 修复 4：PPO update loss NaN 检测
  ★ 修复 5：advantage 归一化 std=0 保护
  ★ 修复 6：优化器 step 前梯度 NaN 检查

新命名：
  代码：train_ppo_40d_mix.py
  模型：best_ppo_40d_mix.pth
  日志：ppo_training_log_40d_mix.txt

用法:
  python train_ppo_40d_mix.py --max-lattices 20 --steps 5000
  python train_ppo_40d_mix.py --steps 300000
"""
import os, glob, math, random, logging, time, argparse, pickle
from collections import deque
import numpy as np
from multiprocessing import Pool, cpu_count
from fractions import Fraction

import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL, BKZ, SVP
except ImportError:
    raise ImportError("请安装 fpylll")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

# ===================== 40维配置 =====================
BASIS_NUM_40 = 40
LATTICE_DIR = "../demo" 
FILE_PATTERN = "random_lattice_40_*"

ACTION_TRIPLE = 0
ACTION_ENUM = 1
ACTION_NOP = 2
NUM_ACTION_TYPES = 3

TRIPLE_REPEAT = 8

START_LLL_DELTA = 0.5
K_SELECT = 28
LLL_FALLBACK_INTERVAL = 50
MAX_STEPS_EPISODE = 100
RESET_UNIMODULAR_MAX_VAL = 3
NO_IMPROVE_STEP_LIMIT = 50

REWARD_START_GAIN = 100.0
REWARD_STEP_GAIN = 500.0
REWARD_INVALID = -0.5
REWARD_TRIPLE_BONUS = 1.0
REWARD_ENUM_COST = -0.5
REWARD_NOP_PENALTY = -0.2

PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR_MAX = 1e-4
LR_MIN = 1e-5
WARMUP_STEPS = 500
ENTROPY_COEF_HI = 0.10
ENTROPY_COEF_LO = 0.05
TOTAL_TRAIN_STEPS = 300_000
NUM_ENVS = 8
BATCH_SIZE = 256
HIDDEN_DIM = 512

K_USE_BKZ = True

GROUND_TRUTH_PKL = "svp_ground_truth_40.pkl"
MODEL_SAVE_PATH = "best_ppo_40d_mix.pth"
LOG_FILE = "ppo_training_log_40d_mix.txt"

STATE_DIM = BASIS_NUM_40 + BASIS_NUM_40 * (BASIS_NUM_40 - 1) // 2 + 2

logging.basicConfig(filename=LOG_FILE, filemode='w',
                    level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler(); console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 工具 =====================
def norm_sq(row):
    return sum(int(x) * int(x) for x in row)


def min_norm_sq(basis):
    return min(norm_sq(row) for row in basis)


def has_zero_row(basis):
    return any(all(int(x) == 0 for x in row) for row in basis)


def sqrt_float_from_sq(nsq):
    if nsq <= 0:
        return 0.0
    return math.exp(0.5 * math.log(nsq))


def load_lattice_40d(file_name, lattice_dir=None):
    if lattice_dir is None:
        lattice_dir = LATTICE_DIR
    fp = os.path.join(lattice_dir, file_name)
    if not os.path.exists(fp):
        for ext in ['.txt', '']:
            cand = os.path.join(lattice_dir, file_name + ext)
            if os.path.exists(cand):
                fp = cand
                break
        else:
            base = file_name.replace('.txt', '')
            all_files = glob.glob(os.path.join(lattice_dir, "*"))
            matched = [f for f in all_files
                       if os.path.basename(f).replace('.txt', '') == base]
            if not matched:
                raise FileNotFoundError(f"未找到 {file_name}")
            fp = matched[0]
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    if n != BASIS_NUM_40:
        raise ValueError(f"{file_name} 不是 40 维")
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


def lll_reduction_safe(basis, delta=0.99):
    n = len(basis)
    m = len(basis[0])
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    try:
        result = LLL.reduction(B_mat, delta=delta)
        B_out = result[0] if isinstance(result, tuple) else result
        new_basis = [[int(B_out[i, j]) for j in range(m)] for i in range(n)]
        if has_zero_row(new_basis):
            return basis
        return new_basis
    except Exception:
        return basis


def gaussian_reduce_2d(u, v, max_iter=50):
    u = [int(x) for x in u]
    v = [int(x) for x in v]
    for _ in range(max_iter):
        dot_uv = sum(u[k] * v[k] for k in range(len(u)))
        norm_u_sq = sum(u[k] * u[k] for k in range(len(u)))
        if norm_u_sq == 0:
            break
        mu = Fraction(dot_uv, norm_u_sq)
        if mu >= 0:
            m = int(mu + Fraction(1, 2))
        else:
            m = -int(-mu + Fraction(1, 2))
        if m == 0 or abs(m) > 10:
            break
        prev_v = v[:]
        v = [v[k] - m * u[k] for k in range(len(v))]
        if all(x == 0 for x in v):
            v = prev_v
            break
    return u, v


def gaussian_reduce_3d(u, v, w, repeat=TRIPLE_REPEAT):
    u = [int(x) for x in u]
    v = [int(x) for x in v]
    w = [int(x) for x in w]
    for _ in range(repeat):
        u, v = gaussian_reduce_2d(u, v)
        u, w = gaussian_reduce_2d(u, w)
        v, w = gaussian_reduce_2d(v, w)
    triple = [u, v, w]
    triple.sort(key=lambda x: sum(int(y) * int(y) for y in x))
    return triple[0], triple[1], triple[2]


def enumerate_sub_lattice(basis_40d, selected_rows):
    k = len(selected_rows)
    if k < 2:
        return None, -1
    try:
        sub_basis = [[int(x) for x in basis_40d[r]] for r in selected_rows]
        B = IntegerMatrix.from_matrix(sub_basis)

        if K_USE_BKZ:
            par = BKZ.Param(block_size=k, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
            BKZ.reduction(B, par)
        else:
            LLL.reduction(B)
            try:
                v = SVP.shortest_vector(B)
                v_int = [int(x) for x in v]
                if len(v_int) != BASIS_NUM_40 or all(x == 0 for x in v_int):
                    return None, -1
                v_sq = sum(x * x for x in v_int)
                if v_sq <= 0:
                    return None, -1
                return v_int, v_sq
            except Exception:
                return None, -1

        best_sq = float('inf')
        best_v = None
        for i in range(k):
            row = [int(B[i, j]) for j in range(BASIS_NUM_40)]
            sq = sum(x * x for x in row)
            if 0 < sq < best_sq:
                best_sq = sq
                best_v = row
        if best_v is None:
            return None, -1
        return best_v, best_sq
    except Exception:
        return None, -1


def compute_state(basis, episode_start_sq, step, max_steps):
    n = BASIS_NUM_40
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]

    cos_vals = []
    for i in range(n):
        row_i = basis[i]
        nsq_i = norm_sqs[i]
        for j in range(i + 1, n):
            row_j = basis[j]
            nsq_j = norm_sqs[j]
            dot = sum(int(row_i[k]) * int(row_j[k]) for k in range(n))
            if dot == 0:
                cos_vals.append(0.0)
                continue
            log_cos_sq = 2 * math.log(abs(dot)) - math.log(nsq_i) - math.log(nsq_j)
            if log_cos_sq > 0:
                log_cos_sq = 0.0
            cos_vals.append(min(abs(math.exp(0.5 * log_cos_sq)), 1.0))

    cur_min = min_norm_sq(basis)
    if episode_start_sq > 0 and cur_min > 0:
        start_ratio = math.log(max(episode_start_sq, 1)) - math.log(max(cur_min, 1))
    else:
        start_ratio = 0.0
    progress = step / max(max_steps, 1)

    state = np.concatenate([
        np.array(log_lengths, dtype=np.float32),
        np.array(cos_vals, dtype=np.float32),
        np.array([start_ratio, progress], dtype=np.float32),
    ])
    state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
    state = np.clip(state, -10.0, 10.0)
    return state.astype(np.float32)


# ===================== ProcessPool worker =====================
def _worker_step(args):
    basis, action_type, selected_rows = args
    prev_basis = [r[:] for r in basis]
    old_min_sq = min_norm_sq(basis)
    zero_incidents = 0

    if action_type == ACTION_TRIPLE:
        if len(selected_rows) >= 3:
            i, j, k = int(selected_rows[0]), int(selected_rows[1]), int(selected_rows[2])
            u, v, w = basis[i], basis[j], basis[k]
            new_u, new_v, new_w = gaussian_reduce_3d(u, v, w)
            basis[i], basis[j], basis[k] = new_u, new_v, new_w
            if has_zero_row(basis):
                basis = prev_basis
                zero_incidents += 1

    elif action_type == ACTION_ENUM:
        v, v_sq = enumerate_sub_lattice(basis, list(selected_rows))
        if (v is not None and v_sq > 0 and v_sq < old_min_sq
                and any(x != 0 for x in v)):
            lens = [norm_sq(r) for r in basis]
            idx_worst = int(np.argmax(lens))
            basis[idx_worst] = v
            if has_zero_row(basis):
                basis = prev_basis
                zero_incidents += 1

    elif action_type == ACTION_NOP:
        pass

    if has_zero_row(basis):
        basis = prev_basis
        zero_incidents += 1

    return basis, zero_incidents


# ===================== 环境 =====================
class SVPEnv40Mix:
    def __init__(self, lat_list):
        if not isinstance(lat_list, list):
            lat_list = [lat_list]
        self.lat_list = lat_list
        self._bind_lattice(random.choice(lat_list))

        self.basis = None
        self.steps = 0
        self.no_improve = 0
        self.best_shortest_sq = 0
        self.best_basis = None
        self.episode_start_sq = 0
        self.zero_row_incidents = 0

    def _bind_lattice(self, info):
        if 'basis' in info and info['basis'] is not None:
            self.orig_basis = info['basis']
        else:
            self.orig_basis = load_lattice_40d(info['file'])

        self.lll_min_sq = int(info['lll_min_sq'])
        self.bkz30_min_sq = int(info.get('bkz30_min_sq', info['lll_min_sq']))
        self.svp_min_sq = int(info.get('svp_min_sq', info['lll_min_sq']))
        self.gh = float(info.get('gh', 0.0))
        self.file_name = info.get('file', '')

    def reset(self, fixed_lattice=None):
        if fixed_lattice is not None:
            self._bind_lattice(fixed_lattice)
        else:
            self._bind_lattice(random.choice(self.lat_list))

        self.basis = lll_reduction_safe(self.orig_basis, delta=START_LLL_DELTA)

        n = BASIS_NUM_40
        U = [[0] * n for _ in range(n)]
        for i in range(n):
            U[i][i] = 1
        for i in range(1, n):
            for j in range(i):
                U[i][j] = random.randint(-RESET_UNIMODULAR_MAX_VAL,
                                          RESET_UNIMODULAR_MAX_VAL + 1)
        new_basis = [
            [sum(int(U[i][k]) * int(self.basis[k][j]) for k in range(n))
             for j in range(n)]
            for i in range(n)
        ]
        if not has_zero_row(new_basis):
            self.basis = new_basis

        self.steps = 0
        self.no_improve = 0
        self.best_shortest_sq = min_norm_sq(self.basis)
        self.best_basis = [r[:] for r in self.basis]
        self.episode_start_sq = self.best_shortest_sq
        self.zero_row_incidents = 0
        return compute_state(self.basis, self.episode_start_sq, 0, MAX_STEPS_EPISODE)

    def apply_step_result(self, new_basis, zero_incidents):
        self.basis = new_basis
        self.zero_row_incidents += zero_incidents
        self.steps += 1

        if LLL_FALLBACK_INTERVAL > 0 and self.steps % LLL_FALLBACK_INTERVAL == 0:
            new_b = lll_reduction_safe(self.basis, delta=0.99)
            if not has_zero_row(new_b):
                self.basis = new_b

        new_min_sq = min_norm_sq(self.basis)
        if new_min_sq < self.best_shortest_sq:
            self.best_shortest_sq = new_min_sq
            self.best_basis = [r[:] for r in self.basis]
            self.no_improve = 0
        else:
            self.no_improve += 1


class PoolEnv40Mix:
    def __init__(self, lat_list, num_envs):
        self.num_envs = num_envs
        self.envs = [SVPEnv40Mix(lat_list) for _ in range(num_envs)]
        self.pool = Pool(processes=min(num_envs, cpu_count()))

    def reset_all(self):
        states = [e.reset() for e in self.envs]
        return np.stack(states, axis=0)

    def step_all(self, action_types, selections):
        args_list = [
            (env.basis, int(at), sel)
            for env, at, sel in zip(self.envs, action_types, selections)
        ]
        results = self.pool.map(_worker_step, args_list)

        s_next = []
        rewards = []
        dones = []
        infos = []

        for i, (env, (new_basis, zero_inc)) in enumerate(zip(self.envs, results)):
            old_min_sq = min_norm_sq(env.basis)
            env.apply_step_result(new_basis, zero_inc)
            new_min_sq = min_norm_sq(env.basis)

            start_len = sqrt_float_from_sq(env.episode_start_sq)
            old_len = sqrt_float_from_sq(old_min_sq)
            new_len = sqrt_float_from_sq(new_min_sq)

            reward = 0.0
            if start_len > 1e-8 and new_len > 1e-8:
                start_gain = math.log(start_len) - math.log(new_len)
                if start_gain > 0:
                    reward += start_gain * REWARD_START_GAIN
            if old_len > 1e-8 and new_len > 1e-8:
                step_gain = math.log(old_len) - math.log(new_len)
                if step_gain > 0:
                    reward += step_gain * REWARD_STEP_GAIN
            if new_min_sq >= old_min_sq:
                reward += REWARD_INVALID

            at = int(action_types[i])
            if at == ACTION_TRIPLE:
                reward += REWARD_TRIPLE_BONUS
            elif at == ACTION_ENUM:
                reward += REWARD_ENUM_COST
            elif at == ACTION_NOP:
                reward += REWARD_NOP_PENALTY

            if env.no_improve > NO_IMPROVE_STEP_LIMIT:
                reward -= 5.0
            reward = float(np.clip(reward, -10.0, 500.0))

            done = env.steps >= MAX_STEPS_EPISODE

            s_next.append(compute_state(env.basis, env.episode_start_sq,
                                         env.steps, MAX_STEPS_EPISODE))
            rewards.append(reward)
            dones.append(done)
            infos.append({
                'cur_shortest': new_len,
                'start_shortest_sq': env.episode_start_sq,
                'lll_min': sqrt_float_from_sq(env.lll_min_sq),
                'bkz30_min': sqrt_float_from_sq(env.bkz30_min_sq),
                'svp_min': sqrt_float_from_sq(env.svp_min_sq),
                'zero_incidents': env.zero_row_incidents,
                'no_improve': env.no_improve,
                'action_type': at,
            })

        return (np.stack(s_next, axis=0),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)

    def reset_single(self, idx):
        return self.envs[idx].reset()

    def close(self):
        self.pool.close()
        self.pool.join()


# ===================== ★ 修复 1：网络输出 nan_to_num =====================
class ActorCritic40Mix(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM),
                                  nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                                  nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.type_head = nn.Linear(HIDDEN_DIM, NUM_ACTION_TYPES)
        self.rows_head = nn.Linear(HIDDEN_DIM, BASIS_NUM_40)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        # 输入保护
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = torch.clamp(x, -10.0, 10.0)
        h = self.fc2(self.fc1(x))
        # 输出保护
        type_logits = self.type_head(h)
        rows_logits = self.rows_head(h)
        value = self.critic(h).squeeze(-1)
        # ★ 关键：logits 和 value 都 nan_to_num
        type_logits = torch.nan_to_num(type_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        rows_logits = torch.nan_to_num(rows_logits, nan=0.0, posinf=10.0, neginf=-10.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
        return type_logits, rows_logits, value


class PPOAgent40Mix:
    def __init__(self):
        self.model = ActorCritic40Mix().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_MAX)

    # ===================== ★ 修复 2：log_probs nan_to_num =====================
    def get_action(self, states, det=False):
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0, posinf=5.0, neginf=-5.0),
                                     -10.0, 10.0)).to(DEVICE)
        with torch.no_grad():
            type_logits, rows_logits, values = self.model(s)

            # 1) action_type 采样
            if det:
                action_types = type_logits.argmax(dim=-1)
            else:
                type_dist = Categorical(logits=type_logits)
                action_types = type_dist.sample()

            # 2) rows Top-K 采样
            if det:
                topk_idx = rows_logits.topk(K_SELECT, dim=-1).indices
            else:
                gumbel = -torch.log(-torch.log(torch.rand_like(rows_logits) + 1e-20) + 1e-20)
                scores = rows_logits + gumbel
                topk_idx = scores.topk(K_SELECT, dim=-1).indices

            # 3) log_prob
            type_logp = F.log_softmax(type_logits, dim=-1).gather(
                1, action_types.unsqueeze(-1)).squeeze(-1)
            rows_logp_all = F.log_softmax(rows_logits, dim=-1)
            rows_logp = rows_logp_all.gather(1, topk_idx).sum(dim=-1)

            # ★ 关键：log_probs nan_to_num
            type_logp = torch.nan_to_num(type_logp, nan=-math.log(NUM_ACTION_TYPES))
            rows_logp = torch.nan_to_num(rows_logp, nan=-math.log(BASIS_NUM_40))
            log_probs = type_logp + rows_logp

        return (action_types.cpu().numpy(),
                topk_idx.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.cpu().numpy())

    # ===================== ★ 修复 3/4/5/6 =====================
    def update(self, states, action_types, selections, old_lp, adv, ret, entropy_coef):
        self.model.train()
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0, posinf=5.0, neginf=-5.0),
                                     -10.0, 10.0)).to(DEVICE)
        at = torch.LongTensor(action_types).to(DEVICE)
        sel = torch.LongTensor(selections).to(DEVICE)
        old = torch.FloatTensor(old_lp).to(DEVICE)
        adv = torch.FloatTensor(adv).to(DEVICE)
        ret = torch.FloatTensor(ret).to(DEVICE)

        # ★ 修复 5：adv 归一化 std=0 保护
        adv_mean = adv.mean()
        adv_std = adv.std()
        if adv_std.item() < 1e-6:
            adv = adv - adv_mean
        else:
            adv = (adv - adv_mean) / (adv_std + EPS)

        # old_lp / adv / ret 也做 nan 保护
        old = torch.nan_to_num(old, nan=0.0)
        adv = torch.nan_to_num(adv, nan=0.0)
        ret = torch.nan_to_num(ret, nan=0.0)

        total_loss = 0
        num = 0
        skipped = 0
        for _ in range(PPO_EPOCHS):
            idxs = np.random.permutation(len(s))
            for st in range(0, len(s), BATCH_SIZE):
                idx = idxs[st:st + BATCH_SIZE]
                type_logits, rows_logits, values = self.model(s[idx])

                # 新的 log_prob
                type_logp = F.log_softmax(type_logits, dim=-1).gather(
                    1, at[idx].unsqueeze(-1)).squeeze(-1)
                rows_logp_all = F.log_softmax(rows_logits, dim=-1)
                rows_logp = rows_logp_all.gather(1, sel[idx]).sum(dim=-1)
                new_logp = type_logp + rows_logp
                new_logp = torch.nan_to_num(new_logp, nan=0.0)

                # 熵
                type_p = F.softmax(type_logits, dim=-1)
                type_ent = -(type_p * (type_p + 1e-10).log()).sum(dim=-1)
                rows_p = F.softmax(rows_logits, dim=-1)
                rows_ent = -(rows_p * (rows_p + 1e-10).log()).sum(dim=-1)
                total_ent = (type_ent + rows_ent).mean()
                total_ent = torch.nan_to_num(total_ent, nan=0.0)

                # ★ 修复 3：ratio clamp(-10, 10)
                diff = torch.clamp(new_logp - old[idx], -10.0, 10.0)
                ratio = torch.exp(diff)

                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret[idx])
                loss = policy_loss + 0.5 * value_loss - entropy_coef * total_ent

                # ★ 修复 4：loss NaN 检测
                if torch.isnan(loss) or torch.isinf(loss):
                    skipped += 1
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                # ★ 修复 6：梯度 NaN 检测
                has_nan_grad = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    skipped += 1
                    self.optimizer.zero_grad()
                    continue

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                num += 1

        if skipped > 0:
            logger.warning(f"  [PPO] 跳过 {skipped} 个 NaN batch")

        return total_loss / max(num, 1)


class RewardRM:
    def __init__(self):
        self.mean, self.var, self.count = 0, 1, 1e-4

    def update(self, x):
        x = np.clip(np.nan_to_num(x, nan=0), -1000, 1000)
        d = np.mean(x) - self.mean
        total = self.count + len(x)
        self.mean += d * len(x) / total
        self.var = max(
            (self.var * self.count + np.var(x) * len(x) + d ** 2 * self.count * len(x) / total) / total,
            EPS)
        self.count = total

    def normalize(self, x):
        return (np.clip(np.nan_to_num(x, nan=0), -1000, 1000) - self.mean) / np.sqrt(self.var + EPS)


def get_lr(step, total):
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    if step < total * 0.8:
        return LR_MAX
    progress = (step - WARMUP_STEPS) / max(total * 0.2, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ===================== 训练器 =====================
class Trainer40Mix:
    def __init__(self, infos, val_infos=None):
        self.infos = infos
        self.val_infos = val_infos if val_infos else infos[:8]
        self.agent = PPOAgent40Mix()
        self.rms = RewardRM()
        self.steps = 0
        self.ep = 0
        self.total_zero_incidents = 0
        self.action_type_counts = {ACTION_TRIPLE: 0, ACTION_ENUM: 0, ACTION_NOP: 0}
        self.start_time = time.time()

    def train(self):
        vec_env = PoolEnv40Mix(self.infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"开始训练 40维 V40-MIX.1（NaN 修复版）！设备：{DEVICE}")
        logger.info(f"K_SELECT={K_SELECT}  TRIPLE_REPEAT={TRIPLE_REPEAT}")
        logger.info(f"NUM_ACTION_TYPES={NUM_ACTION_TYPES}  "
                    f"(0=TRIPLE, 1=ENUM, 2=NOP)")
        logger.info(f"模型保存: {MODEL_SAVE_PATH}")

        try:
            while self.steps < TOTAL_TRAIN_STEPS:
                current_lr = get_lr(self.steps, TOTAL_TRAIN_STEPS)
                for pg in self.agent.optimizer.param_groups:
                    pg['lr'] = current_lr
                progress = self.steps / TOTAL_TRAIN_STEPS
                entropy_coef = ENTROPY_COEF_HI * (1 - progress) + ENTROPY_COEF_LO * progress

                buf_types = []
                buf_sel = []
                buf_states = []
                buf_lp = []
                buf_r = []
                buf_d = []
                buf_v = []
                buf_infos = []

                for step in range(MAX_STEPS_EPISODE):
                    action_types, selections, lp, v = self.agent.get_action(states)
                    current_states = states.copy()
                    s_next, r, d, infos = vec_env.step_all(action_types, selections)

                    for i in range(NUM_ENVS):
                        if d[i]:
                            s_next[i] = vec_env.reset_single(i)

                    buf_states.append(current_states)
                    buf_types.append(action_types)
                    buf_sel.append(selections)
                    buf_lp.append(lp)
                    buf_r.append(r)
                    buf_d.append(d)
                    buf_v.append(v)
                    buf_infos.append(infos)
                    states = s_next
                    self.steps += 1

                    if step % 25 == 0:
                        summary = " | ".join(
                            [f"E{i}:{infos[i]['cur_shortest']:.1f}" for i in range(NUM_ENVS)])
                        elapsed = time.time() - self.start_time
                        speed = self.steps / max(elapsed, 1)
                        type_names = {0: 'T', 1: 'E', 2: 'N'}
                        type_dist = " ".join(
                            [type_names[infos[i]['action_type']] for i in range(NUM_ENVS)])
                        logger.info(f"Step {self.steps} ({speed:.2f}/s) | lr={current_lr:.2e} | "
                                    f"types=[{type_dist}]")
                        logger.info(f"  [{summary}]")
                        b_idx = min(range(NUM_ENVS),
                                    key=lambda i: infos[i].get('cur_shortest', float('inf')))
                        b = infos[b_idx]
                        logger.info(
                            f"  Best E{b_idx} | start={sqrt_float_from_sq(b['start_shortest_sq']):.1f} "
                            f"→ {b['cur_shortest']:.1f} | LLL={b['lll_min']:.1f} | "
                            f"BKZ40={b['bkz30_min']:.1f} | SVP={b['svp_min']:.1f} | "
                            f"zero={b.get('zero_incidents', 0)}")

                _, _, _, last_v = self.agent.get_action(states)
                buf_v.append(last_v)

                T = MAX_STEPS_EPISODE
                all_s = np.stack(buf_states, 0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
                all_at = np.stack(buf_types, 0).transpose(1, 0).reshape(-1)
                all_sel = np.stack(buf_sel, 0).transpose(1, 0, 2).reshape(-1, K_SELECT)
                all_lp = np.stack(buf_lp, 0).transpose(1, 0).reshape(-1)
                all_r = np.stack(buf_r, 0).transpose(1, 0).reshape(-1)
                all_d = np.stack(buf_d, 0).transpose(1, 0).reshape(-1)
                all_v = np.stack(buf_v[:-1], 0).transpose(1, 0).reshape(-1)

                # 动作统计
                for at in all_at:
                    self.action_type_counts[int(at)] += 1

                self.rms.update(all_r)
                all_rn = self.rms.normalize(all_r)

                all_adv, all_ret = np.zeros_like(all_rn), np.zeros_like(all_rn)
                for ei in range(NUM_ENVS):
                    st = ei * T
                    ga = np.zeros(T, dtype=np.float32)
                    last_gae = 0
                    for t in reversed(range(T)):
                        mask = 1 - all_d[st + t]
                        nv = 0 if t == T - 1 else all_v[st + t + 1]
                        delta = all_rn[st + t] + GAMMA * nv * mask - all_v[st + t]
                        last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
                        ga[t] = last_gae
                    all_adv[st:st + T] = ga
                    all_ret[st:st + T] = ga + all_v[st:st + T]

                loss = self.agent.update(all_s, all_at, all_sel, all_lp,
                                          all_adv, all_ret, entropy_coef)
                self.ep += NUM_ENVS

                if self.ep % 200 < NUM_ENVS:
                    cur_zero = sum(inf.get('zero_incidents', 0) for inf in buf_infos[-1])
                    self.total_zero_incidents += cur_zero
                    total_at = sum(self.action_type_counts.values())
                    pct_t = self.action_type_counts[ACTION_TRIPLE] / max(total_at, 1) * 100
                    pct_e = self.action_type_counts[ACTION_ENUM] / max(total_at, 1) * 100
                    pct_n = self.action_type_counts[ACTION_NOP] / max(total_at, 1) * 100
                    logger.info(f"Ep {self.ep} | Loss {loss:.4f} | "
                                f"zero(cum)={self.total_zero_incidents} | "
                                f"动作分布: TRIPLE={pct_t:.1f}% ENUM={pct_e:.1f}% NOP={pct_n:.1f}%")
                    torch.save(self.agent.model.state_dict(), MODEL_SAVE_PATH)
                    logger.info(f"    [Save] 模型已保存: {MODEL_SAVE_PATH}")
        finally:
            vec_env.close()


# ===================== 主程序 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TOTAL_TRAIN_STEPS)
    parser.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_PKL)
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--no-bkz", action="store_true")
    args = parser.parse_args()

    if args.no_bkz:
        K_USE_BKZ = False
        logger.info("[Config] 使用 SVP 枚举（慢）")

    TOTAL_TRAIN_STEPS = args.steps
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if not os.path.exists(args.ground_truth):
        logger.warning(f"真值文件不存在，从原始格文件构建...")
        files = sorted(glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN)))
        if args.max_lattices > 0:
            files = files[:args.max_lattices]
        infos_all = []
        for fp in files:
            try:
                basis = load_lattice_40d(os.path.basename(fp))
                lll_basis = lll_reduction_safe(basis, delta=0.99)
                lll_min_sq = min_norm_sq(lll_basis)
                try:
                    B_mat = IntegerMatrix.from_matrix([[int(x) for x in r] for r in basis])
                    par = BKZ.Param(block_size=40, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
                    BKZ.reduction(B_mat, par)
                    bkz_basis = [[int(B_mat[i, j]) for j in range(40)] for i in range(40)]
                    bkz_min_sq = min_norm_sq(bkz_basis)
                except Exception:
                    bkz_min_sq = lll_min_sq
                infos_all.append({
                    'file': os.path.basename(fp),
                    'basis': basis,
                    'lll_min_sq': int(lll_min_sq),
                    'bkz30_min_sq': int(bkz_min_sq),
                    'svp_min_sq': int(bkz_min_sq),
                    'gh': float('nan'),
                })
                logger.info(f"  {os.path.basename(fp)}: LLL={sqrt_float_from_sq(lll_min_sq):.1f}, "
                            f"BKZ40={sqrt_float_from_sq(bkz_min_sq):.1f}")
            except Exception as e:
                logger.warning(f"  跳过 {fp}: {e}")
    else:
        with open(args.ground_truth, 'rb') as f:
            data = pickle.load(f)
        infos_all = data['lattices']
        if args.max_lattices > 0:
            infos_all = infos_all[:args.max_lattices]
        for info in infos_all:
            if 'basis' not in info or info['basis'] is None:
                try:
                    info['basis'] = load_lattice_40d(info['file'])
                except Exception:
                    info['basis'] = None
        infos_all = [i for i in infos_all if i['basis'] is not None]

    logger.info(f"总格数: {len(infos_all)}")

    if args.skip_filter:
        infos = infos_all
    else:
        infos = [i for i in infos_all if i['svp_min_sq'] < i['lll_min_sq']]
        logger.info(f"[Filter] 过滤后: {len(infos)} / {len(infos_all)}")

    random.shuffle(infos)
    n_val = max(4, len(infos) // 5)
    val_infos = infos[:n_val]
    train_infos = infos[n_val:]
    logger.info(f"训练格: {len(train_infos)}  验证格: {len(val_infos)}")
    logger.info("开始训练...\n")

    trainer = Trainer40Mix(train_infos, val_infos)
    trainer.train()