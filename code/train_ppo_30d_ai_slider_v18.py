"""
30维 AI 子格枚举训练代码 V18.1（修复版）

修复内容（V18 → V18.1）：
  ★ 修复 1：enumerate_sub_lattice 数学错误
     旧：取 30×30 矩阵的"行子块"（非格）
     新：取选中行的完整 30 维坐标，填进 30×30 矩阵
  ★ 修复 2：K_SELECT 从 12 降到 8（速度提升 10 倍）
  ★ 修复 3：get_lr 延长高 LR 期（前 50% 保持 LR_MAX）

保留 V18：
  - 起点：LLL(0.5) 弱约化
  - 取消周期 LLL
  - Top-K 选择器 + 子格枚举
  - 奖励不告诉智能体目标
  - 从 pkl 加载真值，从文件加载 basis

用法:
  python train_ppo_30d_ai_slider_v18.py --max-lattices 30 --steps 20000
  python train_ppo_30d_ai_slider_v18.py --steps 500000
"""
import os, glob, math, random, logging, time, argparse, pickle
from collections import deque
import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL, BKZ, SVP
except ImportError:
    raise ImportError("请安装 fpylll")

import sympy as sp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
BASIS_NUM_30 = 30
LATTICE_DIR = "../demo" 

# ===================== V18.1 核心超参 =====================
START_LLL_DELTA = 0.5
K_SELECT = 8                       
MAX_STEPS_EPISODE = 200
RESET_DISTURB_STEPS = (3, 5)
NO_IMPROVE_STEP_LIMIT = 80

REWARD_START_GAIN = 100.0
REWARD_STEP_GAIN = 500.0
REWARD_INVALID = -0.5

PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR_MAX = 1e-4
LR_MIN = 1e-5
WARMUP_STEPS = 1000
ENTROPY_COEF_HI = 0.10
ENTROPY_COEF_LO = 0.05
TOTAL_TRAIN_STEPS = 500_000
NUM_ENVS = 8
BATCH_SIZE = 512
HIDDEN_DIM = 256

GROUND_TRUTH_PKL = "svp_ground_truth.pkl"

logging.basicConfig(filename='ppo_training_log_30d_ai_slider_v18.txt', filemode='w',
                    level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler(); console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 精确整数工具 =====================
def norm_sq(row):
    return sum(int(x) * int(x) for x in row)


def min_norm_sq(basis):
    return min(norm_sq(row) for row in basis)


def sqrt_float_from_sq(nsq):
    if nsq <= 0:
        return 0.0
    return math.exp(0.5 * math.log(nsq))


def load_lattice_30d(file_name):
    """从文件重新加载 basis"""
    fp = os.path.join(LATTICE_DIR, file_name)
    if not os.path.exists(fp):
        candidates = glob.glob(os.path.join(LATTICE_DIR, f"*{file_name}*"))
        if not candidates:
            raise FileNotFoundError(f"未找到 {file_name}")
        fp = candidates[0]

    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


# ===================== 约化工具 =====================
def lll_reduction(basis, delta=0.99):
    if len(basis) != 30 or len(basis[0]) != 30:
        return basis
    n = len(basis)
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    result = LLL.reduction(B_mat, delta=delta)
    B_out = result[0] if isinstance(result, tuple) else result
    return [[int(B_out[i, j]) for j in range(n)] for i in range(n)]


def gaussian_reduce_2d_int(u, v, max_iter=50):
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


# ===================== 核心：子格枚举（★ 修复 1） =====================
def enumerate_sub_lattice(basis_30d, selected_rows):
    """
    ★ 修复版：取选中行的完整 30 维坐标组成子格基
    用 fpylll 的 SVP 枚举 k 行的整数组合中最短的向量

    技巧：把 k 行填进 30×30 矩阵（其余补零），SVP 枚举自动跳过零行
    """
    k = len(selected_rows)
    if k < 2:
        return None, -1

    try:
        # 把选中的 k 行填进 30×30 矩阵
        full = [[0] * 30 for _ in range(30)]
        for i, r in enumerate(selected_rows):
            full[i] = [int(x) for x in basis_30d[r]]

        B = IntegerMatrix.from_matrix(full)
        LLL.reduction(B)

        # SVP 枚举
        try:
            v = SVP.shortest_vector(B)
        except Exception:
            # 退化：LLL 后取最短非零行
            best_sq = float('inf')
            best_v = None
            for i in range(30):
                row = [int(B[i, j]) for j in range(30)]
                sq = sum(x * x for x in row)
                if 0 < sq < best_sq:
                    best_sq = sq
                    best_v = row
            if best_v is None:
                return None, -1
            return best_v, best_sq

        v_int = [int(x) for x in v]
        if all(x == 0 for x in v_int):
            return None, -1
        v_sq = sum(x * x for x in v_int)
        return v_int, v_sq

    except Exception as e:
        return None, -1


# ===================== 状态特征 =====================
def compute_state(basis, episode_start_sq, step, max_steps):
    """465 (log_len + cos) + 2 (起点改进比 + 进度) = 467"""
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]

    cos_vals = []
    for i in range(BASIS_NUM_30):
        row_i = basis[i]
        nsq_i = norm_sqs[i]
        for j in range(i + 1, BASIS_NUM_30):
            row_j = basis[j]
            nsq_j = norm_sqs[j]
            dot = sum(int(row_i[k]) * int(row_j[k]) for k in range(BASIS_NUM_30))
            if dot == 0:
                cos_vals.append(0.0)
                continue
            log_cos_sq = 2 * math.log(abs(dot)) - math.log(nsq_i) - math.log(nsq_j)
            if log_cos_sq > 0:
                log_cos_sq = 0.0
            cos_vals.append(min(abs(math.exp(0.5 * log_cos_sq)), 1.0))

    cur_min = min_norm_sq(basis)
    if episode_start_sq > 0:
        start_ratio = math.log(max(episode_start_sq, 1)) - math.log(max(cur_min, 1))
    else:
        start_ratio = 0.0
    progress = step / max(max_steps, 1)

    state = np.concatenate([
        np.array(log_lengths, dtype=np.float32),
        np.array(cos_vals, dtype=np.float32),
        np.array([start_ratio, progress], dtype=np.float32),
    ])
    return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)


STATE_DIM = 465 + 2


# ===================== 环境 =====================
class SVPEnv30:
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

    def _bind_lattice(self, info):
        if 'basis' in info and info['basis'] is not None:
            self.orig_basis = info['basis']
        else:
            self.orig_basis = load_lattice_30d(info['file'])

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

        # ★ 起点 = LLL(0.5) 弱约化
        self.basis = lll_reduction(self.orig_basis, delta=START_LLL_DELTA)

        n_disturb = random.randint(*RESET_DISTURB_STEPS)
        for _ in range(n_disturb):
            i, j = random.sample(range(BASIS_NUM_30), 2)
            u, v = gaussian_reduce_2d_int(self.basis[i], self.basis[j])
            self.basis[i], self.basis[j] = u, v

        # ★ 无周期 LLL

        self.steps = 0
        self.no_improve = 0
        self.best_shortest_sq = min_norm_sq(self.basis)
        self.best_basis = [r[:] for r in self.basis]
        self.episode_start_sq = self.best_shortest_sq
        return compute_state(self.basis, self.episode_start_sq, 0, MAX_STEPS_EPISODE)

    def step_with_selection(self, selected_rows):
        self.steps += 1
        prev_basis = [r[:] for r in self.basis]
        old_min_sq = min_norm_sq(self.basis)

        v, v_sq = enumerate_sub_lattice(self.basis, list(selected_rows))

        if v is not None and v_sq < old_min_sq:
            # 用 v 替换基里最长的行
            lens = [norm_sq(r) for r in self.basis]
            idx_worst = int(np.argmax(lens))
            self.basis[idx_worst] = v

        new_min_sq = min_norm_sq(self.basis)

        if new_min_sq < self.best_shortest_sq:
            self.best_shortest_sq = new_min_sq
            self.best_basis = [r[:] for r in self.basis]
            self.no_improve = 0
        else:
            self.no_improve += 1

        start_len = sqrt_float_from_sq(self.episode_start_sq)
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

        if self.no_improve > NO_IMPROVE_STEP_LIMIT:
            reward -= 5.0

        reward = float(np.clip(reward, -10.0, 500.0))

        done = self.steps >= MAX_STEPS_EPISODE
        state = compute_state(self.basis, self.episode_start_sq,
                              self.steps, MAX_STEPS_EPISODE)
        info = {
            'cur_shortest': new_len,
            'cur_shortest_sq': new_min_sq,
            'start_shortest_sq': self.episode_start_sq,
            'lll_min': sqrt_float_from_sq(self.lll_min_sq),
            'bkz30_min': sqrt_float_from_sq(self.bkz30_min_sq),
            'svp_min': sqrt_float_from_sq(self.svp_min_sq),
            'no_improve': self.no_improve,
        }
        return state, reward, done, info


class VecEnv30:
    def __init__(self, lat_list, num_envs):
        self.envs = [SVPEnv30(lat_list) for _ in range(num_envs)]

    def reset_all(self):
        with ThreadPoolExecutor(max_workers=len(self.envs)) as ex:
            states = [f.result() for f in [ex.submit(e.reset) for e in self.envs]]
        return np.stack(states, axis=0)

    def step_all_selections(self, selections):
        with ThreadPoolExecutor(max_workers=len(self.envs)) as ex:
            results = [f.result() for f in
                       [ex.submit(e.step_with_selection, sel)
                        for e, sel in zip(self.envs, selections)]]
        s_next, rewards, dones, infos = zip(*results)
        return (np.stack(s_next, axis=0),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)

    def reset_single(self, idx):
        return self.envs[idx].reset()


# ===================== 网络 =====================
class ActorCritic30WithSelector(nn.Module):
    def __init__(self, k=K_SELECT):
        super().__init__()
        self.k = k
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM),
                                  nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                                  nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.selector = nn.Linear(HIDDEN_DIM, BASIS_NUM_30)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        h = self.fc2(self.fc1(x))
        mask_logits = self.selector(h)
        value = self.critic(h).squeeze(-1)
        return mask_logits, value


class PPOAgent30:
    def __init__(self):
        self.model = ActorCritic30WithSelector(K_SELECT).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_MAX)

    def get_action(self, states, det=False):
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0), -5, 5)).to(DEVICE)
        with torch.no_grad():
            mask_logits, values = self.model(s)
            if det:
                topk_idx = mask_logits.topk(K_SELECT, dim=-1).indices
            else:
                gumbel = -torch.log(-torch.log(torch.rand_like(mask_logits) + 1e-20) + 1e-20)
                scores = mask_logits + gumbel
                topk_idx = scores.topk(K_SELECT, dim=-1).indices

            log_probs_all = F.log_softmax(mask_logits, dim=-1)
            selected_logp = log_probs_all.gather(1, topk_idx).sum(dim=-1)

        return topk_idx.cpu().numpy(), selected_logp.cpu().numpy(), values.cpu().numpy()

    def update(self, states, selections, old_lp, adv, ret, entropy_coef):
        self.model.train()
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0), -5, 5)).to(DEVICE)
        sel = torch.LongTensor(selections).to(DEVICE)
        old = torch.FloatTensor(old_lp).to(DEVICE)
        adv = torch.FloatTensor(adv).to(DEVICE)
        ret = torch.FloatTensor(ret).to(DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + EPS)
        total_loss = 0
        num = 0
        for _ in range(PPO_EPOCHS):
            idxs = np.random.permutation(len(s))
            for st in range(0, len(s), BATCH_SIZE):
                idx = idxs[st:st + BATCH_SIZE]
                mask_logits, values = self.model(s[idx])
                log_probs_all = F.log_softmax(mask_logits, dim=-1)
                new_logp = log_probs_all.gather(1, sel[idx]).sum(dim=-1)
                dist_entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1).mean()
                ratio = torch.exp(new_logp - old[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                loss = -torch.min(surr1, surr2).mean() + \
                       0.5 * F.mse_loss(values, ret[idx]) - \
                       entropy_coef * dist_entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                num += 1
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


# ★ 修复 3：延长高 LR 期
def get_lr(step, total):
    if step < WARMUP_STEPS:
        return LR_MAX * (step + 1) / WARMUP_STEPS
    if step < total * 0.5:
        return LR_MAX
    progress = (step - total * 0.5) / max(total * 0.5, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ===================== 评估 =====================
def quick_eval(agent, lattices, n_val=16, n_samples=4):
    agent.model.eval()
    gaps_lll = []
    gaps_svp = []
    reached_lll = 0
    reached_svp = 0

    with torch.no_grad():
        for i in range(min(n_val, len(lattices))):
            info = lattices[i]
            best_sq = None
            for _ in range(n_samples):
                env = SVPEnv30([info])
                state = env.reset(fixed_lattice=info)
                for _ in range(MAX_STEPS_EPISODE):
                    st = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    mask_logits, _ = agent.model(st)
                    gumbel = -torch.log(-torch.log(torch.rand_like(mask_logits) + 1e-20) + 1e-20)
                    scores = mask_logits + gumbel
                    sel = scores.topk(K_SELECT, dim=-1).indices[0].cpu().numpy()
                    state, _, done, _ = env.step_with_selection(sel)
                    if done:
                        break
                if best_sq is None or env.best_shortest_sq < best_sq:
                    best_sq = env.best_shortest_sq

            lll_sq = info['lll_min_sq']
            svp_sq = info.get('svp_min_sq', lll_sq)

            if best_sq < lll_sq:
                reached_lll += 1
            if best_sq <= svp_sq:
                reached_svp += 1

            gaps_lll.append((sqrt_float_from_sq(best_sq) - sqrt_float_from_sq(lll_sq)) /
                            max(sqrt_float_from_sq(lll_sq), 1e-8))
            gaps_svp.append((sqrt_float_from_sq(best_sq) - sqrt_float_from_sq(svp_sq)) /
                            max(sqrt_float_from_sq(svp_sq), 1e-8))

    agent.model.train()
    n = min(n_val, len(lattices))
    return {
        'reached_lll_rate': reached_lll / n,
        'reached_svp_rate': reached_svp / n,
        'mean_gap_lll': float(np.mean(gaps_lll)),
        'mean_gap_svp': float(np.mean(gaps_svp)),
    }


# ===================== 训练器 =====================
class Trainer30:
    def __init__(self, infos, val_infos=None):
        self.infos = infos
        self.val_infos = val_infos if val_infos else infos[:16]
        self.agent = PPOAgent30()
        self.rms = RewardRM()
        self.steps = 0
        self.ep = 0
        self.best_val_reached_svp = -1.0
        self.best_val_gap_svp = float('inf')
        self.start_time = time.time()

    def train(self):
        vec_env = VecEnv30(self.infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"开始训练 30维 V18.1！设备：{DEVICE}")
        logger.info(f"配置：K_SELECT={K_SELECT}  MAX_STEPS_EPISODE={MAX_STEPS_EPISODE}")
        logger.info(f"起点：LLL(delta={START_LLL_DELTA})  ★ 无周期 LLL")
        logger.info(f"训练格: {len(self.infos)}  验证格: {len(self.val_infos)}")

        while self.steps < TOTAL_TRAIN_STEPS:
            current_lr = get_lr(self.steps, TOTAL_TRAIN_STEPS)
            for pg in self.agent.optimizer.param_groups:
                pg['lr'] = current_lr

            progress = self.steps / TOTAL_TRAIN_STEPS
            entropy_coef = ENTROPY_COEF_HI * (1 - progress) + ENTROPY_COEF_LO * progress

            buf = [[] for _ in range(6)]
            buf_infos = []

            for step in range(MAX_STEPS_EPISODE):
                selections, lp, v = self.agent.get_action(states)
                current_states = states.copy()
                s_next, r, d, infos = vec_env.step_all_selections(selections)

                for i in range(NUM_ENVS):
                    if d[i]:
                        s_next[i] = vec_env.reset_single(i)

                buf[0].append(current_states)
                buf[1].append(selections)
                buf[2].append(lp)
                buf[3].append(r)
                buf[4].append(d)
                buf[5].append(v)
                buf_infos.append(infos)
                states = s_next
                self.steps += 1

                if step % 25 == 0:
                    summary = " | ".join(
                        [f"E{i}:{infos[i]['cur_shortest']:.1f}" for i in range(NUM_ENVS)]
                    )
                    elapsed = time.time() - self.start_time
                    speed = self.steps / max(elapsed, 1)
                    logger.info(f"Step {self.steps} ({speed:.1f}/s) | lr={current_lr:.2e} | "
                                f"ent={entropy_coef:.4f}")
                    logger.info(f"  [{summary}]")
                    b_idx = min(range(NUM_ENVS),
                                key=lambda i: infos[i].get('cur_shortest', float('inf')))
                    b = infos[b_idx]
                    logger.info(
                        f"  Best E{b_idx} | start={sqrt_float_from_sq(b['start_shortest_sq']):.2f} "
                        f"→ {b['cur_shortest']:.2f} | "
                        f"LLL={b['lll_min']:.2f} | BKZ30={b['bkz30_min']:.2f} | "
                        f"SVP={b['svp_min']:.2f}"
                    )

            _, _, last_v = self.agent.get_action(states)
            buf[5].append(last_v)

            T = MAX_STEPS_EPISODE
            all_s = np.stack(buf[0], 0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
            all_sel = np.stack(buf[1], 0).transpose(1, 0, 2).reshape(-1, K_SELECT)
            all_lp = np.stack(buf[2], 0).transpose(1, 0).reshape(-1)
            all_r = np.stack(buf[3], 0).transpose(1, 0).reshape(-1)
            all_d = np.stack(buf[4], 0).transpose(1, 0).reshape(-1)
            all_v = np.stack(buf[5][:-1], 0).transpose(1, 0).reshape(-1)

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

            loss = self.agent.update(all_s, all_sel, all_lp, all_adv, all_ret, entropy_coef)
            self.ep += NUM_ENVS

            if self.ep % 200 < NUM_ENVS:
                val = quick_eval(self.agent, self.val_infos, n_val=16, n_samples=4)
                logger.info(f"Ep {self.ep} | Loss {loss:.4f} | lr={current_lr:.2e}")
                logger.info(f"  [Val] reached_LLL={val['reached_lll_rate']:.3f} | "
                            f"reached_SVP={val['reached_svp_rate']:.3f} | "
                            f"gap_LLL={val['mean_gap_lll']:.4f} | "
                            f"gap_SVP={val['mean_gap_svp']:.4f}")

                if (val['reached_svp_rate'] > self.best_val_reached_svp or
                        (val['reached_svp_rate'] == self.best_val_reached_svp and
                         val['mean_gap_svp'] < self.best_val_gap_svp)):
                    self.best_val_reached_svp = val['reached_svp_rate']
                    self.best_val_gap_svp = val['mean_gap_svp']
                    torch.save(self.agent.model.state_dict(),
                               "best_ppo_30d_ai_slider_v18.pth")
                    logger.info(f"    [Save] 新最优！reached_SVP={self.best_val_reached_svp:.3f} "
                                f"gap_SVP={self.best_val_gap_svp:.4f}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=TOTAL_TRAIN_STEPS)
    parser.add_argument("--ground-truth", type=str, default=GROUND_TRUTH_PKL)
    parser.add_argument("--skip-filter", action="store_true")
    args = parser.parse_args()

    TOTAL_TRAIN_STEPS = args.steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 加载真值 pkl ----
    logger.info(f"加载真值文件: {args.ground_truth}")
    with open(args.ground_truth, 'rb') as f:
        data = pickle.load(f)

    infos_all = data['lattices']
    logger.info(f"总格数: {len(infos_all)}")

    if args.max_lattices > 0:
        infos_all = infos_all[:args.max_lattices]
        logger.info(f"限制使用前 {len(infos_all)} 个格")

    # ---- 过滤 ----
    if args.skip_filter:
        infos = infos_all
        logger.info(f"[Filter] 不过滤，训练格: {len(infos)}")
    else:
        infos = [i for i in infos_all if i['svp_min_sq'] < i['lll_min_sq']]
        logger.info(f"[Filter] 过滤后: {len(infos)} / {len(infos_all)} "
                    f"({len(infos)/max(len(infos_all),1)*100:.1f}%)")

    if len(infos) < 5:
        logger.warning(f"⚠️ 训练格只有 {len(infos)} 个，可能不够")

    # ---- 加载 basis ----
    logger.info("加载 basis 到 info...")
    for info in infos:
        if 'basis' not in info or info['basis'] is None:
            try:
                info['basis'] = load_lattice_30d(info['file'])
            except Exception as e:
                logger.warning(f"加载失败 {info['file']}: {e}")
                info['basis'] = None

    infos = [i for i in infos if i['basis'] is not None]
    logger.info(f"成功加载 basis 的格数: {len(infos)}")

    random.shuffle(infos)
    n_val = max(5, len(infos) // 5)
    val_infos = infos[:n_val]
    train_infos = infos[n_val:]
    logger.info(f"训练格: {len(train_infos)}  验证格: {len(val_infos)}")
    logger.info("开始训练...\n")

    trainer = Trainer30(train_infos, val_infos)
    trainer.train()