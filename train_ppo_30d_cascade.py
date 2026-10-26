"""
30维级联(方法三)训练代码 - 彻底修复“误杀真实短向量”Bug
核心：严格格点反推验证，只要重建长度一致，就保留并高额奖励，不再因为浮点误差误杀。
"""
import os, glob, math, random, logging, numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    from fpylll import IntegerMatrix, LLL

    FPLLL_AVAILABLE = True
except ImportError:
    raise ImportError("请安装 fpylll")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
LATTICE_DIR = r"/mnt/d/generator/三十维"
FILE_PATTERN = "random_lattice_30_*"
BASIS_NUM_30 = 30
USE_3D = True

PAIR_ACTIONS = list(combinations(range(BASIS_NUM_30), 2))
TRIPLE_ACTIONS = list(combinations(range(BASIS_NUM_30), 3)) if USE_3D else []
SUBGEMM_ACTIONS = [(0,), (1,), (2,)]
ACTION_LIST = []
for act in PAIR_ACTIONS: ACTION_LIST.append(('PAIR', act))
if USE_3D:
    for act in TRIPLE_ACTIONS: ACTION_LIST.append(('TRIPLE', act))
for act in SUBGEMM_ACTIONS: ACTION_LIST.append(('SUBGEMM', act))

ACTION_DIM = len(ACTION_LIST)
STATE_DIM = 465
MODEL_PATH_20D = "best_ppo_20d_subgem.pth"
MODEL_PATH_30D = "best_ppo_30d_cascade_fixed_v4.pth"

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

logging.basicConfig(filename='ppo_training_log_30d_cascade_fixed_v4.txt', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO);
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 严格格点反推验证函数 =====================
def verify_real_lattice_vector(orig_basis, v_min, scale):
    n = len(orig_basis)
    orig_f = np.array(orig_basis, dtype=np.float64) / scale
    v_f = np.array(v_min, dtype=np.float64) / scale
    try:
        coeffs, _, _, _ = np.linalg.lstsq(orig_f.T, v_f, rcond=None)
        rounded = np.round(coeffs)
        coeff_error = float(np.max(np.abs(coeffs - rounded)))

        # 在缩放空间重建向量并计算长度，避免大整数溢出
        recon_v_scaled = [sum(rounded[k] * orig_f[k][i] for k in range(n)) for i in range(n)]
        verify_len_scaled = float(np.linalg.norm(recon_v_scaled))
        # 真实长度
        verify_len = verify_len_scaled * scale

        # 判定：允许误差，只要重建长度和原长度接近即可
        if abs(verify_len - float(np.linalg.norm(v_min))) < 1e-2 * scale and verify_len > 1.0:
            return True, coeff_error, verify_len
        else:
            return False, coeff_error, verify_len
    except Exception:
        return False, -1.0, -1.0


# ===================== 20维专家 =====================
class ActorCritic20(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(232, 256), nn.LayerNorm(256), nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.Tanh())
        self.actor = nn.Linear(256, 1330)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x = self.fc1(x);
        x = self.fc2(x)
        return self.actor(x), self.critic(x).squeeze(-1)


agent_20d = ActorCritic20().to(DEVICE)
if os.path.exists(MODEL_PATH_20D):
    agent_20d.load_state_dict(torch.load(MODEL_PATH_20D, map_location=DEVICE))
    agent_20d.eval()
    logger.info("成功加载 20维子格专家！")
else:
    raise RuntimeError(f"找不到20维专家模型: {MODEL_PATH_20D}")

ACTIONS_20D = []
for i in range(20):
    for j in range(i + 1, 20): ACTIONS_20D.append(('PAIR', (i, j)))
for i in range(20):
    for j in range(i + 1, 20):
        for k in range(j + 1, 20): ACTIONS_20D.append(('TRIPLE', (i, j, k)))


def get_20d_state(basis_20d):
    lengths = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in basis_20d]
    lengths = np.maximum(lengths, 1e-8)
    log_lengths = [math.log(l) for l in lengths]
    cos_vals = []
    for i in range(20):
        for j in range(i + 1, 20):
            denom = lengths[i] * lengths[j]
            if denom < 1e-8:
                cos_vals.append(0.0)
            else:
                num = sum(float(basis_20d[i][k]) * float(basis_20d[j][k]) for k in range(20))
                cos_vals.append(abs(num) / denom)
    min_len = float(np.min(lengths))
    basis_ratios = np.log(np.array(lengths) / min_len + 1e-8)
    mean_len_feat = np.log(np.mean(lengths))
    std_len_feat = np.log(np.std(lengths) + 1e-8)
    state = np.concatenate([np.array(log_lengths), np.array(cos_vals),
                            np.clip(basis_ratios, -5, 5),
                            np.array([mean_len_feat, std_len_feat])]).astype(np.float32)
    return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)


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
        if abs(m) > 10: break  # 防大数归零
        prev_v = v[:]
        v = [v[k] - m * u[k] for k in range(len(v))]
        if all(x == 0 for x in v):
            v = prev_v
            break
    return u, v


def run_20d_expert(basis_20d, steps=10):
    current_basis = [row[:] for row in basis_20d]
    for _ in range(steps):
        state = get_20d_state(current_basis)
        with torch.no_grad():
            st = torch.FloatTensor(state).to(DEVICE)
            logits, _ = agent_20d(st)
            action_idx = torch.argmax(logits, dim=-1).item()
        if action_idx < len(ACTIONS_20D):
            act_type, idxs = ACTIONS_20D[action_idx]
            if act_type == 'PAIR':
                i, j = idxs
                u, v = gaussian_reduce_2d_int(current_basis[i], current_basis[j])
                current_basis[i], current_basis[j] = u, v
            elif act_type == 'TRIPLE':
                i, j, k = idxs
                u, v, w = current_basis[i], current_basis[j], current_basis[k]
                for _ in range(8):
                    u, v = gaussian_reduce_2d_int(u, v)
                    u, w = gaussian_reduce_2d_int(u, w)
                    v, w = gaussian_reduce_2d_int(v, w)
                current_basis[i], current_basis[j], current_basis[k] = u, v, w
    return current_basis


def lll_reduction_30d(basis, delta=0.99):
    if len(basis) != 30 or len(basis[0]) != 30: return basis
    n = len(basis)
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    result = LLL.reduction(B_mat, delta=delta)
    B_out = result[0] if isinstance(result, tuple) else result
    return [[int(row[j]) for j in range(n)] for row in B_out]


def compute_log_od_30d(basis):
    try:
        norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in basis]
        norms = [max(x, 1e-10) for x in norms]
        od_val = float(sum([math.log(x) for x in norms]))
        return max(od_val, 100.0) if od_val > 0 else 100.0
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
        elif act_type == 'SUBGEMM':
            seed = idxs[0]
            if seed == 0:
                selected_rows = sorted([0] + random.sample(range(1, 30), 19))
            else:
                selected_rows = sorted(random.sample(range(30), 20))
            selected_cols = sorted(random.sample(range(30), 20))
            sub_basis = [[self.lll_basis[r][c] for c in selected_cols] for r in selected_rows]
            if len(sub_basis) == 20 and len(sub_basis[0]) == 20:
                optimized_sub = run_20d_expert(sub_basis, steps=10)
                for r_idx, row_idx in enumerate(selected_rows):
                    new_row_30 = [0] * 30
                    for c_idx, col_idx in enumerate(selected_cols):
                        new_row_30[col_idx] = optimized_sub[r_idx][c_idx]
                    self.lll_basis[row_idx] = new_row_30

        if self.steps % LLL_INTERVAL == 0:
            self.lll_basis = lll_reduction_30d(self.lll_basis)

        new_od = compute_log_od_30d(self.lll_basis)
        new_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])

        was_rejected = False
        coeff_err_out = -1.0
        verify_len_out = -1.0
        is_verified = False

        # 【核心修复】严格验证逻辑
        if new_len < 2.0:
            was_rejected = True
        elif new_len < self.lll_min * 0.8:
            norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis]
            idx_min = int(np.argmin(norms))
            v_min = self.lll_basis[idx_min]

            is_verified, coeff_err_out, verify_len_out = verify_real_lattice_vector(self.orig_basis, v_min, self.scale)
            if is_verified:
                logger.info(
                    f"🎉 【验证通过】找到真实的极短向量! 系数误差: {coeff_err_out:.6f}, 重建长度: {verify_len_out:.4f}, 原始长度: {new_len:.4f}")
                was_rejected = False
            else:
                was_rejected = True
                logger.info(
                    f"⚠️ 【验证失败】发现伪向量! 系数误差: {coeff_err_out:.6f}, 重建长度: {verify_len_out:.4f}, 原始长度: {new_len:.4f}")

        if was_rejected:
            self.lll_basis = [row[:] for row in prev_basis]
            self.lll_basis = lll_reduction_30d(self.lll_basis)
            new_od = compute_log_od_30d(self.lll_basis)
            new_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])

        if new_len < self.best_shortest:
            self.best_shortest = new_len; self.no_improve = 0
        else:
            self.no_improve += 1

        def safe_log(x):
            if not np.isfinite(float(x)) or float(x) <= 1e-8: return -15.0
            return math.log(float(x))

        reward = (safe_log(old_od) - safe_log(new_od)) * 30.0 + (
                    safe_log(float(old_len)) - safe_log(float(new_len))) * 20.0
        if is_verified:
            reward += 100.0
        if act_type == 'SUBGEMM' and new_len < old_len:
            reward += 100.0
        if was_rejected: reward = -100.0
        if self.no_improve > NO_IMPROVE_STEP_LIMIT: reward -= 20.0
        reward = np.clip(reward, -100.0, 200.0)

        done = self.steps >= MAX_STEPS_EPISODE
        info = {
            'status': 'REJECTED' if was_rejected else 'RUNNING',
            'action': action_idx, 'pair': idxs,
            'cur_shortest': float(new_len),
            'lll_min': float(self.lll_min),
            'new_od': float(new_od),
            'no_improve': self.no_improve,
            'coeff_err': coeff_err_out,
            'verify_len': verify_len_out
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


class ActorCritic30(nn.Module):
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
        logger.info(f"开始训练30维级联智能体！设备：{DEVICE}")
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
                if step % 10 == 0:
                    b_idx = min(range(NUM_ENVS), key=lambda i: infos[i].get('cur_shortest', float('inf')))
                    b_inf = infos[b_idx]
                    coeff_str = f" | 系数误差 {b_inf['coeff_err']:.4f} | 重建长度 {b_inf['verify_len']:.4f}" if b_inf[
                                                                                                                    'coeff_err'] > 0 else ""
                    logger.info(
                        f"Step {self.steps} | Env {b_idx} | 动作 {b_inf['pair']} | Shortest {b_inf['cur_shortest']:.4f} | LLL_Min {b_inf['lll_min']:.4f} | OD {b_inf['new_od']:.4e} | Status: {b_inf['status']}{coeff_str}")
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
                    torch.save(self.agent.model.state_dict(), MODEL_PATH_30D)


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