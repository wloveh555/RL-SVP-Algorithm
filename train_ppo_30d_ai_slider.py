"""
30维训练代码 - AI控制滑窗枚举版
删除子智能体，加入AI控制的前/中/后 5x5 滑窗，调用 BKZ(块大小=5)进行精确枚举。
防伪：纯Python int，防0向量，强制防溢出。
"""
import os, glob, math, random, logging, numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL, BKZ

    FPLLL_AVAILABLE = True
except ImportError:
    raise ImportError("请安装 fpylll")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
LATTICE_DIR = r"/mnt/d/generator/三十维"
FILE_PATTERN = "random_lattice_30_*"
BASIS_NUM_30 = 30
USE_3D = True

# ===================== 动作空间 (删除了 SUBGEMM，引入 AI 滑窗枚举) =====================
PAIR_ACTIONS = list(combinations(range(BASIS_NUM_30), 2))
TRIPLE_ACTIONS = list(combinations(range(BASIS_NUM_30), 3)) if USE_3D else []
SLIDING_WINDOWS = [('BKZ_FRONT', 0), ('BKZ_MIDDLE', 1), ('BKZ_BACK', 2)]  # 前5列/中5列/后5列

ACTION_LIST = []
for act in PAIR_ACTIONS: ACTION_LIST.append(('PAIR', act))
if USE_3D:
    for act in TRIPLE_ACTIONS: ACTION_LIST.append(('TRIPLE', act))
for win_type, idx in SLIDING_WINDOWS: ACTION_LIST.append((win_type, (idx,)))

ACTION_DIM = len(ACTION_LIST)
STATE_DIM = 465

# ===================== 超参数 =====================
PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 1e-4
ENTROPY_COEF = 0.05
MAX_STEPS_EPISODE = 500
TOTAL_TRAIN_STEPS = 200000
NUM_ENVS = 32
LLL_INTERVAL = 10
NO_IMPROVE_STEP_LIMIT = 60
BATCH_SIZE = 1024
HIDDEN_DIM = 256

logging.basicConfig(filename='ppo_training_log_30d_ai_slider.txt', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO);
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 核心数学函数 =====================
def lll_reduction_30d(basis, delta=0.99):
    if len(basis) != 30 or len(basis[0]) != 30: return basis
    n = len(basis)
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    result = LLL.reduction(B_mat, delta=delta)
    B_out = result[0] if isinstance(result, tuple) else result
    return [[int(row[j]) for j in range(n)] for row in B_out]


def gaussian_reduce_2d_int(u, v):
    u = [int(x) for x in u];
    v = [int(x) for x in v]
    for _ in range(50):
        dot_uv = sum(u[k] * v[k] for k in range(len(u)))
        norm_u_sq = sum(u[k] ** 2 for k in range(len(u)))
        if norm_u_sq == 0: break
        mu = dot_uv / norm_u_sq
        if not math.isfinite(mu) or round(mu) == 0: break
        m = int(round(mu))
        if abs(m) > 10: break  # 防止大数灾难性抵消
        prev_v = v[:]
        v = [v[k] - m * u[k] for k in range(len(v))]
        if all(x == 0 for x in v):
            v = prev_v
            break
    return u, v


# ===================== BKZ 滑窗枚举实现 =====================
def run_bkz_on_window(basis_30d, window_type):
    """AI选择窗口后，在局部5x5子矩阵上调用精确枚举（块大小=5）"""
    if window_type == 0:  # 前5列
        rows, cols = list(range(5)), list(range(5))
    elif window_type == 1:  # 中5列
        rows, cols = list(range(5, 10)), list(range(5, 10))
    else:  # 后5列
        rows, cols = list(range(25, 30)), list(range(25, 30))

    # 提取 5x5 子矩阵
    sub_matrix = [[basis_30d[r][c] for c in cols] for r in rows]

    # 调用 fpylll 的 BKZ 精确枚举
    n = 5
    int_sub = IntegerMatrix.from_matrix([[int(x) for x in row] for row in sub_matrix])
    try:
        # 块大小设置为 5，进行极限制枚举（穷举所有可能的组合寻找最短）
        par = BKZ.Param(block_size=5, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
        BKZ.reduction(int_sub, par)

        # 提取优化后的结果
        opt_sub = [[int(int_sub[i, j]) for j in range(n)] for i in range(n)]

        # 回填到原矩阵（确保直接复制，不修改矩阵维度）
        new_basis = [row[:] for row in basis_30d]
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                new_basis[r][c] = opt_sub[i][j]
        return new_basis
    except Exception:
        # 如果 BKZ 因为数据太大无法处理，则直接返回原矩阵（绝对安全）
        return basis_30d


def compute_log_od_30d(basis):
    try:
        norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in basis]
        norms = [max(x, 1e-10) for x in norms]
        od_val = float(sum([math.log(x) for x in norms]))
        return max(od_val, 100.0)  # 防御负值
    except Exception:
        return 100.0


def load_lattice_30d(fp):
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


def preprocess_30d(fp):
    basis_30d = load_lattice_30d(fp)
    try:
        lll_basis = lll_reduction_30d(basis_30d)
        norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in lll_basis]
        lll_min = float(min(norms))
        max_val = max(abs(x) for row in basis_30d for x in row)
        digits = len(str(max_val))
        logger.info(f"提取30维格: {os.path.basename(fp)} | 整数位数: {digits} 位 | LLL_min: {lll_min:.4f}")
        return {'basis': basis_30d, 'lll_min': lll_min, 'gh': lll_min * 1.05, 'scale': 10 ** (digits - 1)}, fp, None
    except Exception:
        return None, fp, "截取失败"


# ===================== 环境定义 =====================
class SVPEnv30:
    def __init__(self, lattice_info, init_len_range=(3.0, 6.0)):
        self.orig_basis = lattice_info['basis']
        self.lll_min = lattice_info['lll_min']
        self.gh = lattice_info['gh']
        self.scale = lattice_info['scale']
        self.n = BASIS_NUM_30
        self.lll_basis = None
        self.steps = 0
        self.no_improve = 0
        self.best_shortest = float('inf')
        self.action_list = ACTION_LIST

    def _get_state_30d(self):
        if self.lll_basis is None: return np.zeros(STATE_DIM, dtype=np.float32)
        lengths = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis]
        lengths = np.maximum(lengths, 1e-8)
        log_lengths = [math.log(l) for l in lengths]
        cos_vals = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                denom = lengths[i] * lengths[j]
                if denom < 1e-8:
                    cos_vals.append(0.0)
                else:
                    num = sum(float(self.lll_basis[i][k]) * float(self.lll_basis[j][k]) for k in range(self.n))
                    cos_vals.append(abs(num) / denom)
        state = np.concatenate([np.array(log_lengths), np.array(cos_vals)]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self):
        U = self.random_unimodular(self.n)
        self.lll_basis = [[sum(self.orig_basis[i][k] * U[k][j] for k in range(self.n)) for j in range(self.n)] for i in
                          range(self.n)]
        self.lll_basis = lll_reduction_30d(self.lll_basis)
        self.steps = 0;
        self.no_improve = 0
        self.best_shortest = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])
        return self._get_state_30d()

    def random_unimodular(self, n, max_val=5):
        L = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: L[i][j] = 1
        for i in range(1, n):
            for j in range(i): L[i][j] = random.randint(-max_val, max_val + 1)
        return L

    def step(self, action_idx):
        self.steps += 1
        act_type, idxs = self.action_list[action_idx][0], self.action_list[action_idx][1]
        prev_basis = [row[:] for row in self.lll_basis]
        old_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])
        old_od = compute_log_od_30d(self.lll_basis)

        # ============ 动作执行 ============
        if act_type == 'PAIR':
            i, j = idxs
            u, v = gaussian_reduce_2d_int(self.lll_basis[i], self.lll_basis[j])
            self.lll_basis[i], self.lll_basis[j] = u, v
        elif act_type == 'TRIPLE':
            i, j, k = idxs
            u, v, w = self.lll_basis[i], self.lll_basis[j], self.lll_basis[k]
            for _ in range(8):
                u, v = gaussian_reduce_2d_int(u, v)
                u, w = gaussian_reduce_2d_int(u, w)
                v, w = gaussian_reduce_2d_int(v, w)
            self.lll_basis[i], self.lll_basis[j], self.lll_basis[k] = u, v, w

        # 【新增动作】：AI控制滑窗选段，然后进行局部 BKZ 枚举
        elif act_type == 'BKZ_FRONT':
            self.lll_basis = run_bkz_on_window(self.lll_basis, 0)
        elif act_type == 'BKZ_MIDDLE':
            self.lll_basis = run_bkz_on_window(self.lll_basis, 1)
        elif act_type == 'BKZ_BACK':
            self.lll_basis = run_bkz_on_window(self.lll_basis, 2)

        # ============ 全局 LLL 兜底 ============
        if self.steps % LLL_INTERVAL == 0:
            self.lll_basis = lll_reduction_30d(self.lll_basis)

        new_od = compute_log_od_30d(self.lll_basis)
        new_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])

        # ============ 物理防线与回滚 ============
        was_rejected = False
        # 如果长度过短（低于2），或者比LLL短太多（低于80%），极有可能是溢出产生的伪向量
        if new_len < 2.0 or new_len < self.lll_min * 0.8:
            was_rejected = True

        if was_rejected:
            self.lll_basis = [row[:] for row in prev_basis]
            self.lll_basis = lll_reduction_30d(self.lll_basis)
            new_od = compute_log_od_30d(self.lll_basis)
            new_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])

        if new_len < self.best_shortest:
            self.best_shortest = new_len; self.no_improve = 0
        else:
            self.no_improve += 1

        # ============ 奖励计算 ============
        def safe_log(x):
            if not np.isfinite(float(x)) or float(x) <= 1e-8: return -15.0
            return math.log(float(x))

        reward = (safe_log(old_od) - safe_log(new_od)) * 30.0 + (
                    safe_log(float(old_len)) - safe_log(float(new_len))) * 20.0
        if was_rejected: reward = -100.0
        if self.no_improve > NO_IMPROVE_STEP_LIMIT: reward -= 20.0
        reward = np.clip(reward, -100.0, 100.0)

        done = self.steps >= MAX_STEPS_EPISODE
        info = {
            'status': 'REJECTED' if was_rejected else 'RUNNING',
            'action': action_idx, 'pair': idxs,
            'cur_shortest': float(new_len),
            'lll_min': float(self.lll_min),
            'new_od': float(new_od),
            'no_improve': self.no_improve
        }
        return self._get_state_30d(), reward, done, info


class VecEnv30:
    def __init__(self, lat_list, num_envs, init_len_range):
        self.envs = [SVPEnv30(random.choice(lat_list), init_len_range) for _ in range(num_envs)]

    def reset_all(self):
        with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
            states = [f.result() for f in [executor.submit(e.reset) for e in self.envs]]
        return np.stack(states, axis=0)

    def step_all(self, actions):
        with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
            results = [f.result() for f in [executor.submit(e.step, a) for e, a in zip(self.envs, actions)]]
        s_next, rewards, dones, infos = zip(*results)
        return np.stack(s_next, axis=0), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), infos

    def reset_single(self, idx):
        return self.envs[idx].reset()


# ===================== PPO 网络与训练器 =====================
class ActorCritic30(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(STATE_DIM, 256), nn.LayerNorm(256), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.Tanh())
        self.actor = nn.Linear(256, ACTION_DIM)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = self.fc1(x);
        x = self.fc2(x)
        return self.actor(x), self.critic(x).squeeze(-1)


class PPOAgent30:
    def __init__(self):
        self.model = ActorCritic30().to(DEVICE)
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


class Trainer30:
    def __init__(self, infos):
        self.infos = infos;
        self.agent = PPOAgent30();
        self.rms = RewardRM()
        self.steps = 0;
        self.ep = 0;
        self.best_od = float('inf')

    def train(self):
        cur_range = (6.0, 10.0)
        vec_env = VecEnv30(self.infos, NUM_ENVS, cur_range)
        states = vec_env.reset_all()
        logger.info(f"开始训练30维 AI滑窗BKZ智能体！设备：{DEVICE}")
        while self.steps < TOTAL_TRAIN_STEPS:
            if self.ep > 8000: cur_range = (3.0, 5.0)
            if self.ep > 16000: cur_range = (2.5, 4.0)
            if self.ep % 1000 == 0: vec_env = VecEnv30(self.infos, NUM_ENVS, cur_range); states = vec_env.reset_all()
            buf = [[] for _ in range(6)];
            buf_infos = []
            for step in range(MAX_STEPS_EPISODE):
                actions, lp, v = self.agent.get_action(states)
                s_next, r, d, infos = vec_env.step_all(actions)
                for i in range(NUM_ENVS):
                    if infos[i].get('no_improve', 0) >= NO_IMPROVE_STEP_LIMIT:
                        d[i] = True;
                        r[i] -= 20
                        states[i] = vec_env.reset_single(i)
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
                        f"Step {self.steps} | Env {b_idx} | 动作 {b_inf['pair']} | Shortest {b_inf['cur_shortest']:.4f} | LLL_Min {b_inf['lll_min']:.4f} | Status: {b_inf['status']}")
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
                logger.info(f"Ep {self.ep} | Loss {loss:.4f} | BestShortest {info_0.get('cur_shortest', 0):.2e}")
                if info_0.get('cur_shortest', float('inf')) < self.best_od:
                    self.best_od = info_0.get('cur_shortest')
                    torch.save(self.agent.model.state_dict(), "best_ppo_30d_ai_slider.pth")


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
    logger.info(f"找到 {len(files)} 个30维格，开始预处理...")

    with Pool(min(cpu_count(), len(files))) as pool:
        res = pool.map(preprocess_30d, files[:limit])
    infos = [r[0] for r in res if r[2] is None]
    if not infos: raise RuntimeError("无法提取任何合格的30维格！")
    logger.info(f"成功提取 {len(infos)} 个30维格。开始训练...")
    trainer = Trainer30(infos);
    trainer.train()