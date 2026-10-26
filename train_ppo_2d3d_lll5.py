"""
PPO 训练智能体 - 2D+3D 混合动作 + LLL周期5
纯净修复版：去除高级特征，严格保持210维状态，修复所有底层Bug。
"""
import os, glob, math, random, logging, numpy as np
from collections import deque
from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL

    FPLLL_AVAILABLE = True
except ImportError:
    raise ImportError("请安装 fpylll: pip install fpylll")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
GH = 1268.98
LATTICE_DIR = r"/mnt/d/generator/二十维"
FILE_PATTERN = "random_lattice_20_*"

# ===================== 超参数 (严格保持 210 维) =====================
USE_3D = True
ACTION_DIM = 1330 if USE_3D else 190
STATE_DIM = 210  # 20长度 + 190夹角
BASIS_NUM = 20

PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
ENTROPY_COEF = 0.05
MAX_STEPS_EPISODE = 500
TOTAL_TRAIN_STEPS = 500000
NUM_ENVS = 32
LLL_INTERVAL = 5
NO_IMPROVE_STEP_LIMIT = 60
BATCH_SIZE = 1024
HIDDEN_DIM = 256

logging.basicConfig(filename='ppo_training_log.txt', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO);
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 基础工具 (无缺失函数) =====================
def lll_reduction(basis, delta=0.99):
    n = basis.shape[0]
    int_basis_list = [[int(round(basis[i, j])) for j in range(n)] for i in range(n)]
    B_mat = IntegerMatrix.from_matrix(int_basis_list)
    result = LLL.reduction(B_mat, delta=delta)
    B_out = result[0] if isinstance(result, tuple) else result
    B_final = np.array([list(row) for row in B_out], dtype=np.float64)
    if B_final.ndim == 1: B_final = B_final.reshape(n, n)
    return B_final, np.eye(n)


def gaussian_reduce_2d(u, v, max_iter=20):
    u, v = u.copy().astype(np.float64), v.copy().astype(np.float64)
    T = np.eye(2, dtype=object)
    it = 0
    while it < max_iter:
        it += 1
        if np.dot(u, u) > np.dot(v, v): u, v = v, u; T = T[:, [1, 0]]
        norm_u_sq = np.dot(u, u) + EPS
        mu = np.dot(v, u) / norm_u_sq
        if not np.isfinite(mu) or round(mu) == 0: break
        m = round(mu);
        v -= m * u;
        T[:, 1] -= m * T[:, 0]
    return u, v, T


def gaussian_reduce_3d(u, v, w):
    u, v, w = u.copy(), v.copy(), w.copy()
    for _ in range(8):
        u, v, _ = gaussian_reduce_2d(u, v)
        u, w, _ = gaussian_reduce_2d(u, w)
        v, w, _ = gaussian_reduce_2d(v, w)
    vecs = [u, v, w];
    norms = [np.linalg.norm(x) for x in vecs];
    order = np.argsort(norms)
    return vecs[order[0]], vecs[order[1]], vecs[order[2]]


def compute_ortho_def(basis, logdet):
    try:
        norms = np.maximum(np.linalg.norm(basis, axis=1), EPS)
        log_od = np.sum(np.log(norms)) - logdet
        return np.exp(log_od) if log_od <= 700 else np.inf
    except Exception:
        return np.inf


def load_lattice(fp):
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = [float(x) for x in content.split() if x]
    n = int(round(math.sqrt(len(nums))))
    return np.array(nums, dtype=np.float64).reshape(n, n)


def random_unimodular(n, max_val=5):
    L = np.eye(n, dtype=object)
    for i in range(1, n):
        for j in range(i): L[i, j] = np.random.randint(-max_val, max_val + 1)
    perm = np.random.permutation(n)
    P = np.zeros((n, n), dtype=np.int64);
    P[np.arange(n), perm] = 1
    U_obj = L @ P.astype(object);
    U = U_obj.astype(np.float64)
    try:
        return U if abs(round(np.linalg.det(U))) == 1 else np.eye(n, dtype=np.float64)
    except Exception:
        return np.eye(n, dtype=np.float64)


def preprocess(fp):
    basis = load_lattice(fp)
    sign, logdet = np.linalg.slogdet(basis)
    if sign == 0: return None, fp, "Singular"
    lll_basis, _ = lll_reduction(basis)
    lll_min = float(np.min(np.linalg.norm(lll_basis, axis=1)))
    return {'basis': basis, 'logdet': logdet, 'lll_min': lll_min}, fp, None


# ===================== 环境定义 (210维) =====================
class SVPEnv:
    def __init__(self, lattice_info, init_len_range=(3.0, 6.0)):
        self.orig_basis = lattice_info['basis']
        self.logdet = lattice_info['logdet']
        self.lll_min = lattice_info['lll_min']
        self.init_len_range = init_len_range
        self.n = BASIS_NUM
        self.lll_basis = None
        self.lll_norms = None
        self.coeffs = None
        self.cur_len = None
        self.best_shortest = float('inf')
        self.best_od = float('inf')
        self.steps = 0
        self.no_improve = 0
        self.history_len = deque(maxlen=8)  # 补齐初始化
        self.action_list = self._build_actions()
        assert len(self.action_list) == ACTION_DIM

    def _build_actions(self):
        acts = []
        for i in range(self.n):
            for j in range(i + 1, self.n): acts.append(('PAIR', i, j))
        if USE_3D:
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    for k in range(j + 1, self.n): acts.append(('TRIPLE', i, j, k))
        return acts

    def _get_state(self):
        if np.any(np.isnan(self.lll_basis)) or np.any(np.isinf(self.lll_basis)):
            return np.zeros(STATE_DIM, dtype=np.float32)

        # 严格的 210 维特征 (20 + 190)
        lengths = np.linalg.norm(self.lll_basis, axis=1) + EPS
        log_lengths = np.log(lengths)
        cos_vals = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                cos_vals.append(
                    np.clip(np.dot(self.lll_basis[i], self.lll_basis[j]) / (lengths[i] * lengths[j]), -1.0, 1.0))

        state = np.concatenate([log_lengths, np.array(cos_vals)]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self):
        U = random_unimodular(self.n)
        self.lll_basis = self.orig_basis @ U
        for i in range(self.n):
            for j in range(i + 1, self.n):
                v1, v2, _ = gaussian_reduce_2d(self.lll_basis[i], self.lll_basis[j])
                self.lll_basis[i], self.lll_basis[j] = v1, v2
        self.lll_basis = self.lll_basis.astype(np.float64)
        self.lll_basis, _ = lll_reduction(self.lll_basis)
        self.lll_norms = np.linalg.norm(self.lll_basis, axis=1)

        lo, hi = self.init_len_range
        target_len = self.lll_min * random.uniform(lo, hi)
        self.coeffs = np.zeros(self.n, dtype=np.int64)
        for _ in range(20):
            c = np.random.randint(-2, 3, size=self.n).astype(np.int64)
            if np.all(c == 0): continue
            l = np.linalg.norm(c.astype(np.float64) @ self.lll_basis)
            if l > 0:
                mult = max(1, int(round(target_len / l)))
                if np.max(np.abs(c * mult)) < 1e9:
                    self.coeffs = c * mult;
                    break
        if np.all(self.coeffs == 0): self.coeffs[np.argmin(self.lll_norms)] = 1

        self.cur_len = float(np.linalg.norm(self.coeffs.astype(np.float64) @ self.lll_basis))
        self.best_shortest = self.cur_len
        self.best_od = compute_ortho_def(self.lll_basis, self.logdet)
        self.steps = 0;
        self.no_improve = 0;
        self.history_len.clear()
        return self._get_state()

    def step(self, action_idx):
        self.steps += 1
        act_type, idxs = self.action_list[action_idx][0], self.action_list[action_idx][1:]
        prev_basis = self.lll_basis.copy()
        prev_coeffs = self.coeffs.copy()

        old_od = compute_ortho_def(self.lll_basis, self.logdet)
        old_len = float(np.min(np.linalg.norm(self.lll_basis, axis=1) + EPS))

        if act_type == 'PAIR':
            i, j = idxs
            v1, v2, _ = gaussian_reduce_2d(self.lll_basis[i], self.lll_basis[j])
            self.lll_basis[i], self.lll_basis[j] = v1, v2
        elif act_type == 'TRIPLE':
            i, j, k = idxs
            v1, v2, v3 = gaussian_reduce_3d(self.lll_basis[i], self.lll_basis[j], self.lll_basis[k])
            self.lll_basis[i], self.lll_basis[j], self.lll_basis[k] = v1, v2, v3

        self.cur_len = float(np.linalg.norm(self.coeffs.astype(np.float64) @ self.lll_basis))

        if self.steps % LLL_INTERVAL == 0:
            self.lll_basis, _ = lll_reduction(self.lll_basis)

        new_od = compute_ortho_def(self.lll_basis, self.logdet)
        new_len = float(np.min(np.linalg.norm(self.lll_basis, axis=1) + EPS))

        was_rejected = False
        # 拦截非物理值和假零
        if not np.isfinite(new_od) or new_od > 1e50 or new_len <= 1e-3 or new_len > self.lll_min * 500:
            self.lll_basis = prev_basis
            self.coeffs = prev_coeffs
            self.lll_basis, _ = lll_reduction(self.lll_basis)
            new_od = compute_ortho_def(self.lll_basis, self.logdet)
            new_len = float(np.min(np.linalg.norm(self.lll_basis, axis=1) + EPS))
            was_rejected = True

        if new_od < self.best_od:
            self.best_od = new_od; self.no_improve = 0
        elif new_len < self.best_shortest:
            self.best_shortest = new_len; self.no_improve = 0
        else:
            self.no_improve += 1

        def safe_log(x):
            if not np.isfinite(x) or x <= 1e-8: return -15.0
            return math.log(x)

        reward = (safe_log(old_od) - safe_log(new_od)) * 30.0 + (safe_log(old_len) - safe_log(new_len)) * 20.0
        if new_len < self.lll_min: reward += 100.0
        if was_rejected: reward = -100.0
        if self.no_improve > NO_IMPROVE_STEP_LIMIT: reward -= 20.0
        reward = np.clip(reward, -100.0, 100.0)

        done = self.steps >= MAX_STEPS_EPISODE
        info = {
            'status': 'REJECTED' if was_rejected else 'RUNNING', 'action': action_idx, 'pair': idxs,
            'cur_shortest': new_len, 'lll_min': self.lll_min, 'new_od': new_od, 'no_improve': self.no_improve
        }
        return self._get_state(), reward, done, info


# ===================== 向量化环境 (修复 Tuple 问题) =====================
class VecEnv:
    def __init__(self, lat_list, num_envs, init_len_range):
        self.envs = [SVPEnv(random.choice(lat_list), init_len_range) for _ in range(num_envs)]

    def reset_all(self):
        with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
            states = [f.result() for f in [executor.submit(e.reset) for e in self.envs]]
        return np.stack(states, axis=0)

    def step_all(self, actions):
        with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
            results = [f.result() for f in [executor.submit(e.step, a) for e, a in zip(self.envs, actions)]]
        s_next, rewards, dones, infos = zip(*results)
        # 修复核心：转成可修改的 numpy 数组
        return np.stack(s_next, axis=0), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), infos

    def reset_single(self, idx):
        return self.envs[idx].reset()


# ===================== PPO 网络与训练器 =====================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Tanh())
        self.actor = nn.Linear(HIDDEN_DIM, ACTION_DIM)
        self.critic = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = self.fc1(x);
        x = self.fc2(x)
        return self.actor(x), self.critic(x).squeeze(-1)


class PPOAgent:
    def __init__(self):
        self.model = ActorCritic().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.entropy = ENTROPY_COEF

    def get_action(self, states, det=False):
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0), -5, 5)).to(DEVICE)
        with torch.no_grad():
            logits, values = self.model(s)
            dist = Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if det else dist.sample()
            log_probs = torch.nan_to_num(dist.log_prob(actions), nan=-math.log(ACTION_DIM))
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, states, actions, old_lp, adv, ret):
        self.model.train()
        s = torch.from_numpy(np.clip(np.nan_to_num(states, nan=0.0), -5, 5)).to(DEVICE)
        a = torch.LongTensor(actions).to(DEVICE)
        old = torch.FloatTensor(old_lp).to(DEVICE)
        adv = torch.FloatTensor(adv).to(DEVICE)
        ret = torch.FloatTensor(ret).to(DEVICE)
        adv = (adv - adv.mean()) / (adv.std() + EPS)
        total_loss = 0;
        num = 0
        for _ in range(PPO_EPOCHS):
            idxs = np.random.permutation(len(s))
            for st in range(0, len(s), BATCH_SIZE):
                idx = idxs[st:st + BATCH_SIZE]
                logits, values = self.model(s[idx])
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(a[idx]);
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - old[idx])
                surr1 = ratio * adv[idx];
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(values, ret[idx]) - self.entropy * entropy
                self.optimizer.zero_grad();
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step();
                total_loss += loss.item();
                num += 1
        return total_loss / max(num, 1)


class RewardRM:
    def __init__(self): self.mean, self.var, self.count = 0, 1, 1e-4

    def update(self, x):
        x = np.clip(np.nan_to_num(x, nan=0), -1000, 1000)
        d = np.mean(x) - self.mean;
        total = self.count + len(x)
        self.mean += d * len(x) / total
        self.var = max((self.var * self.count + np.var(x) * len(x) + d ** 2 * self.count * len(x) / total) / total, EPS)
        self.count = total

    def normalize(self, x): return (np.clip(np.nan_to_num(x, nan=0), -1000, 1000) - self.mean) / np.sqrt(self.var + EPS)


class Trainer:
    def __init__(self, infos):
        self.infos = infos;
        self.agent = PPOAgent();
        self.rms = RewardRM()
        self.steps = 0;
        self.ep = 0;
        self.best_od = float('inf')

    def train(self):
        cur_range = (6.0, 10.0)
        vec_env = VecEnv(self.infos, NUM_ENVS, cur_range)
        states = vec_env.reset_all()

        while self.steps < TOTAL_TRAIN_STEPS:
            if self.ep > 8000: cur_range = (3.0, 5.0)
            if self.ep > 16000: cur_range = (2.5, 4.0)
            if self.ep % 1000 == 0: vec_env = VecEnv(self.infos, NUM_ENVS, cur_range); states = vec_env.reset_all()

            buf = [[] for _ in range(6)];
            buf_infos = []
            for step in range(MAX_STEPS_EPISODE):
                actions, lp, v = self.agent.get_action(states)
                s_next, r, d, infos = vec_env.step_all(actions)

                # 现在 d 是 numpy 数组，可以顺利修改了
                for i in range(NUM_ENVS):
                    if infos[i].get('no_improve', 0) >= NO_IMPROVE_STEP_LIMIT or infos[i].get('cur_shortest', 0) > 1000:
                        d[i] = True;
                        r[i] -= 20
                        states[i] = vec_env.reset_single(i)
                if all(infos[i].get('cur_shortest', float('inf')) > 1000 for i in range(NUM_ENVS)):
                    states = vec_env.reset_all()

                buf[0].append(states);
                buf[1].append(actions);
                buf[2].append(lp)
                buf[3].append(r);
                buf[4].append(d);
                buf[5].append(v)
                buf_infos.append(infos)
                states = s_next;
                self.steps += 1

                if step % 50 == 0:
                    b_idx = min(range(NUM_ENVS), key=lambda i: infos[i].get('cur_shortest', float('inf')))
                    b_inf = infos[b_idx]
                    logger.info(
                        f"Step {self.steps} | Env {b_idx} | Status {b_inf['status']} | Shortest {b_inf['cur_shortest']:.2e} | LLL_Min {b_inf['lll_min']:.4f} | OD {b_inf['new_od']:.2e}")

            _, _, last_v = self.agent.get_action(states)
            buf[5].append(last_v)
            T = MAX_STEPS_EPISODE
            all_s = np.stack(buf[0], 0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
            all_a = np.stack(buf[1], 0).transpose(1, 0).reshape(-1)
            all_lp = np.stack(buf[2], 0).transpose(1, 0).reshape(-1)
            all_r = np.stack(buf[3], 0).transpose(1, 0).reshape(-1)
            all_d = np.stack(buf[4], 0).transpose(1, 0).reshape(-1)
            all_v = np.stack(buf[5][:-1], 0).transpose(1, 0).reshape(-1)

            self.rms.update(all_r);
            all_rn = self.rms.normalize(all_r)
            all_adv, all_ret = np.zeros_like(all_rn), np.zeros_like(all_rn)
            for ei in range(NUM_ENVS):
                st = ei * T;
                ga = np.zeros(T, dtype=np.float32);
                last_gae = 0
                for t in reversed(range(T)):
                    mask = 1 - all_d[st + t]
                    nv = 0 if t == T - 1 else all_v[st + t + 1]
                    delta = all_rn[st + t] + GAMMA * nv * mask - all_v[st + t]
                    last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
                    ga[t] = last_gae
                all_adv[st:st + T] = ga;
                all_ret[st:st + T] = ga + all_v[st:st + T]

            loss = self.agent.update(all_s, all_a, all_lp, all_adv, all_ret)
            self.ep += NUM_ENVS
            if self.ep % 100 < NUM_ENVS:
                info_0 = buf_infos[-1][0]
                logger.info(f"Ep {self.ep} | Loss {loss:.4f} | BestOD {info_0.get('new_od', 0):.2e}")
                if info_0.get('new_od', float('inf')) < self.best_od:
                    self.best_od = info_0.get('new_od')
                    torch.save(self.agent.model.state_dict(), "best_ppo_od_model.pth")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)

    files = glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN))
    if not files: raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}")
    limit = len(files) if args.max_lattices <= 0 else min(args.max_lattices, len(files))
    logger.info(f"找到 {len(files)} 个格，预处理...")
    with Pool(min(cpu_count(), len(files))) as pool:
        res = pool.map(preprocess, files[:limit])
    infos = [r[0] for r in res if r[2] is None]
    if not infos: raise RuntimeError("无可用格文件！")
    logger.info(f"加载 {len(infos)} 个格子。开始训练...")
    trainer = Trainer(infos);
    trainer.train()