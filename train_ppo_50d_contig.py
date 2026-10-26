"""
50维 连续窗口训练代码 V50-CONTIG.2.1

V50-CONTIG.2 → V50-CONTIG.2.1 的唯一修复：
  ★ 修复：过滤逻辑。因为真值 = LLL(0.99)（svp_min_sq == lll_min_sq），
    "svp < lll" 永不成立 → 过滤掉全部格。改为默认不过滤。

保留 V50-CONTIG.2 全部机制：
  - START_LLL_DELTA = 0.7
  - MAX_STEPS_EPISODE = 200
  - LLL_FALLBACK_INTERVAL = 100
  - NO_IMPROVE_STEP_LIMIT = 100
  - 真值 = LLL(0.99)

命名：
  代码：train_ppo_50d_contig_v2.py
  模型：best_ppo_50d_contig_v2.pth
  日志：ppo_training_log_50d_contig_v2.txt

用法:
  python train_ppo_50d_contig_v2.py --max-lattices 20 --steps 5000
  python train_ppo_50d_contig_v2.py --steps 300000
  python train_ppo_50d_contig_v2.py --with-bkz45 --max-lattices 20 --steps 5000
"""
import os, glob, math, random, logging, time, argparse, pickle
import numpy as np
from multiprocessing import Pool, cpu_count

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

# ===================== 50维配置 V2.1 =====================
BASIS_NUM_50 = 50
LATTICE_DIR = r"/mnt/d/generator/五十维"
FILE_PATTERN = "random_lattice_50_*"

START_LLL_DELTA = 0.7
K_SELECT = 35
LLL_FALLBACK_INTERVAL = 100
MAX_STEPS_EPISODE = 200
RESET_UNIMODULAR_MAX_VAL = 3
NO_IMPROVE_STEP_LIMIT = 100

REWARD_START_GAIN = 100.0
REWARD_STEP_GAIN = 500.0
REWARD_INVALID = -0.5

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
HIDDEN_DIM = 640

K_USE_BKZ = True

GROUND_TRUTH_PKL = "svp_ground_truth_50.pkl"
MODEL_SAVE_PATH = "best_ppo_50d_contig_v2.pth"
LOG_FILE = "ppo_training_log_50d_contig_v2.txt"

STATE_DIM = BASIS_NUM_50 + BASIS_NUM_50 * (BASIS_NUM_50 - 1) // 2 + 2  # 1277

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


def load_lattice_50d(file_name, lattice_dir=None):
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
    if n != BASIS_NUM_50:
        raise ValueError(f"{file_name} 不是 50 维 (解析为 {n})")
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


# ===================== 约化工具 =====================
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


def enumerate_sub_lattice(basis_50d, selected_rows):
    k = len(selected_rows)
    if k < 2:
        return None, -1
    try:
        sub_basis = [[int(x) for x in basis_50d[r]] for r in selected_rows]
        B = IntegerMatrix.from_matrix(sub_basis)

        if K_USE_BKZ:
            par = BKZ.Param(block_size=k, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
            BKZ.reduction(B, par)
        else:
            LLL.reduction(B)
            try:
                v = SVP.shortest_vector(B)
                v_int = [int(x) for x in v]
                if len(v_int) != BASIS_NUM_50 or all(x == 0 for x in v_int):
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
            row = [int(B[i, j]) for j in range(BASIS_NUM_50)]
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
    n = BASIS_NUM_50
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
    return np.clip(state, -10.0, 10.0).astype(np.float32)


# ===================== ProcessPool worker =====================
def _worker_step(args):
    basis, selected_rows = args
    prev_basis = [r[:] for r in basis]
    old_min_sq = min_norm_sq(basis)
    zero_incidents = 0

    v, v_sq = enumerate_sub_lattice(basis, list(selected_rows))
    if (v is not None and v_sq > 0 and v_sq < old_min_sq
            and any(x != 0 for x in v)):
        lens = [norm_sq(r) for r in basis]
        idx_worst = int(np.argmax(lens))
        basis[idx_worst] = v
        if has_zero_row(basis):
            basis = prev_basis
            zero_incidents += 1

    if has_zero_row(basis):
        basis = prev_basis
        zero_incidents += 1

    return basis, zero_incidents


# ===================== 环境 =====================
class SVPEnv50:
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
            self.orig_basis = load_lattice_50d(info['file'])

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

        n = BASIS_NUM_50
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


class PoolEnv50:
    def __init__(self, lat_list, num_envs):
        self.num_envs = num_envs
        self.envs = [SVPEnv50(lat_list) for _ in range(num_envs)]
        self.pool = Pool(processes=min(num_envs, cpu_count()))

    def reset_all(self):
        states = [e.reset() for e in self.envs]
        return np.stack(states, axis=0)

    def step_all(self, selections):
        args_list = [(env.basis, sel) for env, sel in zip(self.envs, selections)]
        results = self.pool.map(_worker_step, args_list)

        s_next, rewards, dones, infos = [], [], [], []

        for i, (env, (new_basis, zero_inc)) in enumerate(zip(self.envs, results)):
            old_min_sq = min_norm_sq(env.basis)
            env.apply_step_result(new_basis, zero_inc)
            new_min_sq = min_norm_sq(env.basis)

            start_len = sqrt_float_from_sq(env.episode_start_sq)
            old_len = sqrt_float_from_sq(old_min_sq)
            new_len = sqrt_float_from_sq(new_min_sq)

            reward = 0.0
            if start_len > 1e-8 and new_len > 1e-8:
                sg = math.log(start_len) - math.log(new_len)
                if sg > 0:
                    reward += sg * REWARD_START_GAIN
            if old_len > 1e-8 and new_len > 1e-8:
                sg = math.log(old_len) - math.log(new_len)
                if sg > 0:
                    reward += sg * REWARD_STEP_GAIN
            if new_min_sq >= old_min_sq:
                reward += REWARD_INVALID
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


# ===================== 网络 =====================
class ActorCritic50(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM),
                                  nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                                  nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.selector = nn.Linear(HIDDEN_DIM, BASIS_NUM_50)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = torch.clamp(x, -10.0, 10.0)
        h = self.fc2(self.fc1(x))
        rows_logits = torch.nan_to_num(self.selector(h), nan=0.0,
                                        posinf=10.0, neginf=-10.0)
        value = torch.nan_to_num(self.critic(h).squeeze(-1), nan=0.0,
                                  posinf=100.0, neginf=-100.0)
        return rows_logits, value


class PPOAgent50:
    def __init__(self):
        self.model = ActorCritic50().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_MAX)

    def get_action(self, states, det=False):
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0),
                                     -10.0, 10.0)).to(DEVICE)
        with torch.no_grad():
            rows_logits, values = self.model(s)

            if det:
                start_idx = rows_logits.argmax(dim=-1)
            else:
                start_dist = Categorical(logits=rows_logits)
                start_idx = start_dist.sample()

            B = start_idx.shape[0]
            arange_k = torch.arange(K_SELECT, device=DEVICE).unsqueeze(0)
            selected = (start_idx.unsqueeze(1) + arange_k) % BASIS_NUM_50

            log_probs = F.log_softmax(rows_logits, dim=-1).gather(
                1, start_idx.unsqueeze(-1)).squeeze(-1)
            log_probs = torch.nan_to_num(log_probs, nan=-math.log(BASIS_NUM_50))

        return (selected.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.cpu().numpy())

    def update(self, states, selections, old_lp, adv, ret, entropy_coef):
        self.model.train()
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0),
                                     -10.0, 10.0)).to(DEVICE)
        sel = torch.LongTensor(selections).to(DEVICE)
        old = torch.FloatTensor(old_lp).to(DEVICE)
        adv = torch.FloatTensor(adv).to(DEVICE)
        ret = torch.FloatTensor(ret).to(DEVICE)

        adv_mean = adv.mean()
        adv_std = adv.std()
        if adv_std.item() < 1e-6:
            adv = adv - adv_mean
        else:
            adv = (adv - adv_mean) / (adv_std + EPS)
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
                rows_logits, values = self.model(s[idx])

                start_idx = sel[idx][:, 0]
                new_logp = F.log_softmax(rows_logits, dim=-1).gather(
                    1, start_idx.unsqueeze(-1)).squeeze(-1)
                new_logp = torch.nan_to_num(new_logp, nan=0.0)

                p = F.softmax(rows_logits, dim=-1)
                ent = -(p * (p + 1e-10).log()).sum(dim=-1).mean()
                ent = torch.nan_to_num(ent, nan=0.0)

                diff = torch.clamp(new_logp - old[idx], -10.0, 10.0)
                ratio = torch.exp(diff)
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret[idx])
                loss = policy_loss + 0.5 * value_loss - entropy_coef * ent

                if torch.isnan(loss) or torch.isinf(loss):
                    skipped += 1
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                has_nan_grad = False
                for p_ in self.model.parameters():
                    if p_.grad is not None and (torch.isnan(p_.grad).any()
                                                 or torch.isinf(p_.grad).any()):
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
        self.var = max((self.var * self.count + np.var(x) * len(x)
                         + d ** 2 * self.count * len(x) / total) / total, EPS)
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
class Trainer50:
    def __init__(self, infos, val_infos):
        self.infos = infos
        self.val_infos = val_infos
        self.agent = PPOAgent50()
        self.rms = RewardRM()
        self.steps = 0
        self.ep = 0
        self.total_zero_incidents = 0
        self.start_time = time.time()

    def train(self):
        vec_env = PoolEnv50(self.infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"开始训练 50维 V50-CONTIG.2.1！设备：{DEVICE}")
        logger.info(f"K_SELECT={K_SELECT}  LLL_INTERVAL={LLL_FALLBACK_INTERVAL}  "
                    f"MAX_STEPS={MAX_STEPS_EPISODE}  START_LLL_DELTA={START_LLL_DELTA}")
        logger.info(f"STATE_DIM={STATE_DIM}  HIDDEN_DIM={HIDDEN_DIM}")
        logger.info(f"模型保存: {MODEL_SAVE_PATH}")

        try:
            while self.steps < TOTAL_TRAIN_STEPS:
                current_lr = get_lr(self.steps, TOTAL_TRAIN_STEPS)
                for pg in self.agent.optimizer.param_groups:
                    pg['lr'] = current_lr
                progress = self.steps / TOTAL_TRAIN_STEPS
                entropy_coef = ENTROPY_COEF_HI * (1 - progress) + ENTROPY_COEF_LO * progress

                buf_s, buf_sel, buf_lp, buf_r, buf_d, buf_v, buf_i = (
                    [], [], [], [], [], [], [])

                for step in range(MAX_STEPS_EPISODE):
                    selections, lp, v = self.agent.get_action(states)
                    current_states = states.copy()
                    s_next, r, d, infos = vec_env.step_all(selections)

                    for i in range(NUM_ENVS):
                        if d[i]:
                            s_next[i] = vec_env.reset_single(i)

                    buf_s.append(current_states)
                    buf_sel.append(selections)
                    buf_lp.append(lp)
                    buf_r.append(r)
                    buf_d.append(d)
                    buf_v.append(v)
                    buf_i.append(infos)
                    states = s_next
                    self.steps += 1

                    if step % 50 == 0:
                        summary = " | ".join(
                            [f"E{i}:{infos[i]['cur_shortest']:.1f}" for i in range(NUM_ENVS)])
                        elapsed = time.time() - self.start_time
                        speed = self.steps / max(elapsed, 1)
                        starts = [int(sel[0]) for sel in selections]
                        logger.info(f"Step {self.steps} ({speed:.2f}/s) | lr={current_lr:.2e} | "
                                    f"starts={starts}")
                        logger.info(f"  [{summary}]")
                        b_idx = min(range(NUM_ENVS),
                                    key=lambda i: infos[i].get('cur_shortest', float('inf')))
                        b = infos[b_idx]
                        logger.info(
                            f"  Best E{b_idx} | start={sqrt_float_from_sq(b['start_shortest_sq']):.1f} "
                            f"→ {b['cur_shortest']:.1f} | LLL={b['lll_min']:.1f} | "
                            f"BKZ50={b['bkz30_min']:.1f} | SVP={b['svp_min']:.1f}")

                _, _, last_v = self.agent.get_action(states)
                buf_v.append(last_v)

                T = MAX_STEPS_EPISODE
                all_s = np.stack(buf_s, 0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
                all_sel = np.stack(buf_sel, 0).transpose(1, 0, 2).reshape(-1, K_SELECT)
                all_lp = np.stack(buf_lp, 0).transpose(1, 0).reshape(-1)
                all_r = np.stack(buf_r, 0).transpose(1, 0).reshape(-1)
                all_d = np.stack(buf_d, 0).transpose(1, 0).reshape(-1)
                all_v = np.stack(buf_v[:-1], 0).transpose(1, 0).reshape(-1)

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

                loss = self.agent.update(all_s, all_sel, all_lp,
                                          all_adv, all_ret, entropy_coef)
                self.ep += NUM_ENVS

                if self.ep % 200 < NUM_ENVS:
                    cur_zero = sum(inf.get('zero_incidents', 0) for inf in buf_i[-1])
                    self.total_zero_incidents += cur_zero
                    logger.info(f"Ep {self.ep} | Loss {loss:.4f} | "
                                f"zero(cum)={self.total_zero_incidents}")
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
    parser.add_argument("--with-bkz45", action="store_true",
                        help="额外用 BKZ(45) 精细化真值（慢）")
    args = parser.parse_args()

    if args.no_bkz:
        K_USE_BKZ = False
        logger.info("[Config] 使用 SVP 枚举（慢）")

    TOTAL_TRAIN_STEPS = args.steps
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    if not os.path.exists(args.ground_truth):
        logger.warning(f"真值文件不存在，从原始格文件构建...")
        files = sorted(glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN)))
        if not files:
            raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}/{FILE_PATTERN}")
        if args.max_lattices > 0:
            files = files[:args.max_lattices]
        infos_all = []
        for fp in files:
            try:
                basis = load_lattice_50d(os.path.basename(fp))

                # 真值 = LLL(0.99)
                lll_basis = lll_reduction_safe(basis, delta=0.99)
                lll_min_sq = min_norm_sq(lll_basis)

                # 可选 BKZ(45) 精细化
                bkz_min_sq = lll_min_sq
                if args.with_bkz45:
                    try:
                        B_mat = IntegerMatrix.from_matrix(
                            [[int(x) for x in r] for r in basis])
                        par = BKZ.Param(block_size=45, strategies=BKZ.DEFAULT_STRATEGY,
                                        max_loops=1)
                        BKZ.reduction(B_mat, par)
                        bkz_basis = [[int(B_mat[i, j]) for j in range(50)]
                                     for i in range(50)]
                        bkz_min_sq = min_norm_sq(bkz_basis)
                    except Exception:
                        pass

                infos_all.append({
                    'file': os.path.basename(fp),
                    'basis': basis,
                    'lll_min_sq': int(lll_min_sq),
                    'bkz30_min_sq': int(bkz_min_sq),
                    'svp_min_sq': int(bkz_min_sq),
                    'gh': float('nan'),
                })
                logger.info(f"  {os.path.basename(fp)}: "
                            f"LLL(0.99)={sqrt_float_from_sq(lll_min_sq):.1f}, "
                            f"BKZ45={sqrt_float_from_sq(bkz_min_sq):.1f}")
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
                    info['basis'] = load_lattice_50d(info['file'])
                except Exception:
                    info['basis'] = None
        infos_all = [i for i in infos_all if i['basis'] is not None]

    logger.info(f"总格数: {len(infos_all)}")

    # ★★★ 核心修复：过滤逻辑 ★★★
    # 默认不过滤（因为真值 = LLL(0.99) 时 svp_min_sq == lll_min_sq）
    # 只有当 --with-bkz45 且 BKZ(45) 真的改进了真值时才过滤
    if args.skip_filter:
        infos = infos_all
        logger.info(f"[Filter] 跳过过滤，训练格: {len(infos)}")
    elif args.with_bkz45:
        infos = [i for i in infos_all if i['svp_min_sq'] < i['lll_min_sq']]
        if len(infos) < 5:
            logger.warning(f"⚠️ 过滤后只有 {len(infos)} 个格，改为不过滤")
            infos = infos_all
        logger.info(f"[Filter] 使用 BKZ45 真值过滤后: {len(infos)} / {len(infos_all)}")
    else:
        # ★ 关键修复：默认不过滤
        infos = infos_all
        logger.info(f"[Filter] 未使用 BKZ45 精细真值，不过滤: {len(infos)} 个格")

    if len(infos) == 0:
        raise RuntimeError("⚠️ 没有可用的格！请检查 LATTICE_DIR 路径是否正确")

    random.shuffle(infos)
    n_val = max(4, len(infos) // 5)
    val_infos = infos[:n_val]
    train_infos = infos[n_val:]

    if len(train_infos) == 0:
        # 如果格太少，全部用于训练（验证用训练集代替）
        train_infos = infos
        val_infos = infos[:min(4, len(infos))]
        logger.warning(f"⚠️ 训练格不足，全部用于训练")

    logger.info(f"训练格: {len(train_infos)}  验证格: {len(val_infos)}")
    logger.info("开始训练...\n")

    trainer = Trainer50(train_infos, val_infos)
    trainer.train()