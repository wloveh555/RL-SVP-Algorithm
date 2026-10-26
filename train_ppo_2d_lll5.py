"""
PPO 训练智能体 - 纯 2D 动作 + LLL 周期 5 (NumPy 矩阵加速 + 5080 适配)
目标：最小化正交性缺陷 (Orthogonality Defect)
核心优化：使用矩阵乘法替代 for 循环计算夹角，速度提升 10 倍以上。
"""
import os
import sys
import glob
import math
import random
import argparse
import logging
import numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ===================== 核心依赖 fpylll =====================
try:
    from fpylll import IntegerMatrix, LLL

    FPLLL_AVAILABLE = True
except ImportError:
    raise ImportError("无法导入 fpylll，请确保已安装：pip install fpylll")

# -------------------- 基础配置与加速设置 --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

EPS = 1e-8
GH = 1268.98

# 实验配置
LATTICE_DIR = r"/mnt/d/generator/二十维"
FILE_PATTERN = "random_lattice_20_*"
ACTION_DIM = 190  # 纯2D动作
STATE_DIM = 210  # 20 + 190
BASIS_NUM = 20

# PPO 超参数 (匹配高速并行)
PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
ENTROPY_COEF = 0.01
MAX_STEPS_EPISODE = 500
TOTAL_TRAIN_STEPS = 500000
EVAL_INTERVAL = 5000
HIDDEN_DIM = 256

# 【重点】根据 RTX 5080 的巨大算力和显存，默认直接开启 64 个并行环境！
NUM_ENVS = 64
LLL_INTERVAL = 5
LOG_STEP_INTERVAL = 10  # 每10步记录一次详细数据

# 批处理大小随环境数量动态增大
BATCH_SIZE = 1024 if NUM_ENVS >= 32 else 512

# -------------------- 日志配置 --------------------
logging.basicConfig(
    filename='ppo_training_log.txt',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 数学工具与环境类 =====================
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


def gaussian_reduce_2d(u, v, max_iter=100):
    u, v = u.copy().astype(np.float64), v.copy().astype(np.float64)
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        if np.dot(u, u) > np.dot(v, v):
            u, v = v, u
        norm_u_sq = np.dot(u, u) + EPS
        mu = np.dot(v, u) / norm_u_sq
        if not np.isfinite(mu): break
        m = round(mu)
        if m == 0: break
        v -= m * u
    return u, v


def compute_ortho_def(basis, logdet):
    try:
        norms = np.linalg.norm(basis, axis=1)
        norms = np.maximum(norms, EPS)
        log_norm_sum = np.sum(np.log(norms))
        log_od = log_norm_sum - logdet
        if not np.isfinite(log_od) or log_od > 700:
            return np.inf
        return np.exp(log_od)
    except Exception:
        return np.inf


def load_lattice(filepath):
    with open(filepath, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = [float(x) for x in content.split() if x]
    n = int(round(math.sqrt(len(nums))))
    return np.array(nums, dtype=np.float64).reshape(n, n)


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


def preprocess_lattice(fp):
    try:
        basis = load_lattice(fp)
        sign, logdet = np.linalg.slogdet(basis)
        if sign == 0: return None, fp, "Singular"
        lll_basis, _ = lll_reduction(basis)
        lll_min = float(np.min(np.linalg.norm(lll_basis, axis=1)))
        info = {
            'basis': basis, 'logdet': logdet,
            'lll_min': lll_min, 'filename': os.path.basename(fp)
        }
        return info, fp, None
    except Exception as e:
        return None, fp, str(e)


# ===================== 环境定义 =====================
class SVPEnv:
    def __init__(self, lattice_info):
        self.orig_basis = lattice_info['basis']
        self.n = BASIS_NUM
        self.logdet = lattice_info['logdet']
        self.lll_min = lattice_info['lll_min']
        self.best_ortho_def = float('inf')
        self.basis = None
        self.steps = 0
        self.action_list = list(combinations(range(self.n), 2))
        assert len(self.action_list) == 190

    def _get_state(self):
        if np.any(np.isnan(self.basis)) or np.any(np.isinf(self.basis)):
            return np.zeros(STATE_DIM, dtype=np.float32)
        lengths = np.linalg.norm(self.basis, axis=1) + EPS
        log_lengths = np.log(lengths)
        cos_vals = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cos_val = np.dot(self.basis[i], self.basis[j]) / (lengths[i] * lengths[j])
                cos_vals.append(np.clip(cos_val, -1.0, 1.0))
        state = np.concatenate([log_lengths, np.array(cos_vals)]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self):
        U = random_unimodular(self.n)
        self.basis = self.orig_basis @ U
        for i in range(self.n):
            for j in range(i + 1, self.n):
                v1, v2 = gaussian_reduce_2d(self.basis[i], self.basis[j])
                self.basis[i], self.basis[j] = v1, v2
        self.basis = self.basis.astype(np.float64)
        self.steps = 0
        self.best_ortho_def = compute_ortho_def(self.basis, self.logdet)
        return self._get_state()

    def step(self, action_idx):
        self.steps += 1
        old_od = compute_ortho_def(self.basis, self.logdet)
        i, j = self.action_list[action_idx]
        v1, v2 = gaussian_reduce_2d(self.basis[i], self.basis[j])
        self.basis[i], self.basis[j] = v1, v2
        if self.steps % LLL_INTERVAL == 0:
            self.basis, _ = lll_reduction(self.basis)
        if np.any(np.isnan(self.basis)) or np.any(np.isinf(self.basis)):
            return self._get_state(), -100.0, True, {'old_od': old_od, 'new_od': float('inf')}

        new_od = compute_ortho_def(self.basis, self.logdet)
        is_best = new_od < self.best_ortho_def
        if is_best: self.best_ortho_def = new_od

        # ============= 【优化重点】使用 NumPy 矩阵运算秒杀双重循环 =============
        lengths = np.linalg.norm(self.basis, axis=1) + EPS
        cur_shortest = float(np.min(lengths))  # 当前最短向量

        # 矩阵运算一次性算出所有夹角！
        G = self.basis @ self.basis.T
        outer_prod = np.outer(lengths, lengths) + EPS
        cos_matrix = np.abs(G) / outer_prod
        np.fill_diagonal(cos_matrix, 0.0)  # 对角线（自身夹角）置 0

        max_cos = float(np.max(cos_matrix))
        mean_cos = float(np.mean(cos_matrix))

        # =====================================================================

        # 奖励计算
        def safe_log_od(od):
            if not np.isfinite(od) or od <= 1e-8: return np.log(1e-8)
            return np.log(od)

        improvement = safe_log_od(old_od) - safe_log_od(new_od)
        reward = np.clip(improvement * 50.0, -30.0, 50.0)
        if is_best: reward += 20.0

        done = self.steps >= MAX_STEPS_EPISODE
        info = {
            'action': action_idx, 'pair': (i, j),
            'cur_shortest': cur_shortest, 'lll_min': self.lll_min,
            'max_cos': max_cos, 'mean_cos': mean_cos,
            'new_od': new_od, 'is_best': is_best
        }
        return self._get_state(), reward, done, info


# ===================== 向量化环境 =====================
class VecEnv:
    def __init__(self, lattice_list, num_envs):
        self.num_envs = num_envs
        self.lattice_list = lattice_list
        self.envs = [SVPEnv(random.choice(lattice_list)) for _ in range(num_envs)]

    def reset_all(self):
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(env.reset) for env in self.envs]
            states = [f.result() for f in futures]
        return np.stack(states, axis=0)

    def step_all(self, actions):
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(env.step, a) for env, a in zip(self.envs, actions)]
            results = [f.result() for f in futures]
        s_next, rewards, dones, infos = zip(*results)
        return np.stack(s_next, axis=0), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), infos


# ===================== GAE 优势计算函数 =====================
def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        next_val = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values[:T]


# ===================== PPO 网络与智能体 =====================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.actor = nn.Linear(HIDDEN_DIM, ACTION_DIM)
        self.critic = nn.Linear(HIDDEN_DIM, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.actor(x)
        values = self.critic(x).squeeze(-1)
        return logits, values


class PPOAgent:
    def __init__(self):
        self.model = ActorCritic().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.entropy_coef = ENTROPY_COEF

    def get_action(self, states_np, deterministic=False):
        states_np = np.clip(np.nan_to_num(states_np, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        states = torch.from_numpy(states_np).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            logits, values = self.model(states)
            dist = Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_probs = dist.log_prob(actions)
            log_probs = torch.nan_to_num(log_probs, nan=-math.log(ACTION_DIM))
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, states, actions, old_log_probs, advantages, returns):
        self.model.train()
        states = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)).to(
            DEVICE, non_blocking=True)
        actions = torch.from_numpy(actions).to(DEVICE, non_blocking=True).long()
        old_log_probs = torch.from_numpy(old_log_probs).to(DEVICE, non_blocking=True)
        advantages = torch.from_numpy(advantages).to(DEVICE, non_blocking=True)
        returns = torch.from_numpy(returns).to(DEVICE, non_blocking=True)

        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
        total_loss, num_batches = 0.0, 0
        indices = np.arange(len(states))

        for _ in range(PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(states), BATCH_SIZE):
                batch_idx = indices[start:start + BATCH_SIZE]
                b_s, b_a, b_old_lp, b_adv, b_ret = states[batch_idx], actions[batch_idx], old_log_probs[batch_idx], \
                advantages[batch_idx], returns[batch_idx]
                self.optimizer.zero_grad()
                with torch.amp.autocast('cuda'):  # RTX 5080 Tensor Cores 高速混合精度
                    logits, values = self.model(b_s)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(b_a)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(log_probs - b_old_lp)
                    surr = -torch.min(ratio * b_adv, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv).mean()
                    value_loss = F.mse_loss(values, b_ret)
                    loss = surr + 0.5 * value_loss - self.entropy_coef * entropy
                if torch.isnan(loss).any() or torch.isinf(loss).any(): continue
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        return total_loss / max(num_batches, 1)


# ===================== 奖励归一化与训练器 =====================
class RewardRunningMeanStd:
    def __init__(self): self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.clip(np.nan_to_num(x, nan=0.0), -1000.0, 1000.0)
        delta = np.mean(x) - self.mean
        total = self.count + len(x)
        self.mean += delta * len(x) / total
        self.var = max((self.var * self.count + np.var(x) * len(x) + delta ** 2 * self.count * len(x) / total) / total,
                       EPS)
        self.count = total

    def normalize(self, x): return (np.clip(np.nan_to_num(x, nan=0.0), -1000.0, 1000.0) - self.mean) / np.sqrt(
        self.var + EPS)


class Trainer:
    def __init__(self, lattice_infos):
        self.lattice_infos = lattice_infos
        self.agent = PPOAgent()
        self.reward_rms = RewardRunningMeanStd()
        self.total_steps = 0
        self.episodes_done = 0
        self.best_od = float('inf')
        logger.info(f"RTX 5080 极速版训练启动！{NUM_ENVS}个环境并行，目标：最小化正交缺陷。LLL周期：{LLL_INTERVAL}。")
        logger.info(f"加载 {len(lattice_infos)} 个格文件。设备: {DEVICE}")

    def train(self):
        vec_env = VecEnv(self.lattice_infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"{NUM_ENVS}个环境并行启动成功。由于矩阵运算加速，训练会极快。")

        while self.total_steps < TOTAL_TRAIN_STEPS:
            # 每个 Batch 收集 N*MAX_STEPS 条轨迹！
            buf_s, buf_a, buf_lp, buf_r, buf_d, buf_v, buf_infos = [], [], [], [], [], [], []

            for step in range(MAX_STEPS_EPISODE):
                actions, log_probs, values = self.agent.get_action(states)
                s_next, rewards, dones, infos = vec_env.step_all(actions)

                # ===== 【保留科研所需打印】（取第0号环境为代表，防止刷屏死机） =====
                if step % LOG_STEP_INTERVAL == 0:
                    first_info = infos[0]
                    logger.info(
                        f"Step {self.total_steps} | Env 0 | "
                        f"Action {first_info['action']:3d} ({first_info['pair']}) | "
                        f"CurShortest {first_info['cur_shortest']:.4f} | "
                        f"LLL_Min {first_info['lll_min']:.4f} | "
                        f"MaxCos {first_info['max_cos']:.4f} | "
                        f"MeanCos {first_info['mean_cos']:.4f} | "
                        f"OrthoDef {first_info['new_od']:.2e}"
                    )
                # =================================================================

                buf_s.append(states)
                buf_a.append(actions)
                buf_lp.append(log_probs)
                buf_r.append(rewards)
                buf_d.append(dones)
                buf_v.append(values)
                buf_infos.append(infos)

                states = s_next
                self.total_steps += 1

            _, _, last_values = self.agent.get_action(states)
            buf_v.append(last_values)

            # 数据重塑以大Batch更新
            T = MAX_STEPS_EPISODE
            all_s = np.stack(buf_s, axis=0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
            all_a = np.stack(buf_a, axis=0).transpose(1, 0).reshape(-1)
            all_lp = np.stack(buf_lp, axis=0).transpose(1, 0).reshape(-1)
            all_r = np.stack(buf_r, axis=0).transpose(1, 0).reshape(-1)
            all_d = np.stack(buf_d, axis=0).transpose(1, 0).reshape(-1)
            all_v = np.stack(buf_v[:-1], axis=0).transpose(1, 0).reshape(-1)

            # 奖励归一化
            self.reward_rms.update(all_r)
            all_r_norm = self.reward_rms.normalize(all_r)

            all_adv, all_ret = np.zeros_like(all_r_norm), np.zeros_like(all_r_norm)
            for env_i in range(NUM_ENVS):
                start = env_i * T
                adv, ret = compute_gae(all_r_norm[start:start + T], all_v[start:start + T], all_d[start:start + T],
                                       GAMMA, GAE_LAMBDA)
                all_adv[start:start + T], all_ret[start:start + T] = adv, ret

            loss = self.agent.update(all_s, all_a, all_lp, all_adv, all_ret)
            self.episodes_done += NUM_ENVS

            # 宏观指标打印
            if self.episodes_done % (NUM_ENVS * 10) < NUM_ENVS:
                ep_mean_rewards = [np.sum([buf_r[i][env] for i in range(T)]) for env in range(NUM_ENVS)]
                info_0 = buf_infos[-1][0]
                logger.info(
                    f"Ep {self.episodes_done:5d} | Step {self.total_steps:7d} | Loss {loss:.4f} | "
                    f"AvgR {np.mean(ep_mean_rewards):7.2f} | BestOD {info_0.get('new_od', 0):.2e}"
                )

            if self.total_steps % 5000 < MAX_STEPS_EPISODE:
                if info_0.get('new_od', float('inf')) < self.best_od:
                    self.best_od = info_0.get('new_od')
                    torch.save(self.agent.model.state_dict(), "best_ppo_od_model.pth")
                    logger.info(f"=== Step {self.total_steps}: 新最佳模型 (OD={self.best_od:.2e}) ===")


# ===================== 主程序 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=0, help="限制加载的格数量，默认0表示加载全部")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files = glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN))
    if not files: raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}")

    limit = len(files) if args.max_lattices <= 0 else min(args.max_lattices, len(files))
    logger.info(f"找到 {len(files)} 个格，预处理...")

    with Pool(min(cpu_count(), len(files))) as pool:
        results = pool.map(preprocess_lattice, files[:limit])

    all_infos = []
    for info, fname, err in results:
        if err:
            logger.warning(f"跳过 {fname}: {err}")
        else:
            all_infos.append(info)
            logger.info(f"[PPO Base] 文件: {fname} | LLL_min: {info['lll_min']:.4f}")

    if not all_infos: raise RuntimeError("无可用格文件！")

    logger.info(f"\n加载完毕！共 {len(all_infos)} 个格子。开始 {NUM_ENVS} 环境并行训练...")
    trainer = Trainer(all_infos)
    trainer.train()