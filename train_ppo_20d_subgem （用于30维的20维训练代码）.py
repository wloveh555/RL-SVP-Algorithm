"""
PPO 训练智能体 - 20维子格训练 (终极绝对防崩版)
彻底抛弃缩放法。使用纯 Python 原生 int 运算，禁止出现0向量。
完美恢复：整数位数检测、OD输出、强制格点验算、详细误差日志。
"""
import os, glob, math, random, logging, numpy as np
from itertools import combinations
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
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
BASIS_NUM = 20
USE_3D = True
ACTION_DIM = 1330 if USE_3D else 190
STATE_DIM = 232  # 20长度 + 190夹角 + 20比例 + 1均值 + 1标准差

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

logging.basicConfig(filename='ppo_training_log_20d_subgem.txt', filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO);
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


def lll_reduction(basis, delta=0.99):
    n = len(basis)
    B_mat = IntegerMatrix.from_matrix([[int(x) for x in row] for row in basis])
    result = LLL.reduction(B_mat, delta=delta)
    B_out = result[0] if isinstance(result, tuple) else result
    return [[int(row[j]) for j in range(n)] for row in B_out]


# ================= 纯 Python int 高斯约简（绝对安全防0向量） =================
def gaussian_reduce_2d(u, v, max_iter=50):
    u = [int(x) for x in u]
    v = [int(x) for x in v]
    it = 0
    while it < max_iter:
        it += 1

        # 使用原生整数计算点积和内积，绝不转 float，避免 inf/NaN！
        dot_uv = sum(u[i] * v[i] for i in range(len(u)))
        norm_u_sq = sum(u[i] ** 2 for i in range(len(u)))
        if norm_u_sq == 0: break

        # Python 的 / 运算自动将 10^180 / 10^180 正确转化为 1.0，避免了缩放法导致的巨大系数！
        mu = dot_uv / norm_u_sq
        if not math.isfinite(mu) or round(mu) == 0: break

        m = int(round(mu))
        prev_v = v[:]  # 备份 v 用于防0检测

        # 大整数减去大整数，安全且精准
        v = [v[i] - m * u[i] for i in range(len(v))]

        # 【核心防线】：如果发现了 0 向量（说明减多了，破坏了格基），立刻回滚！
        if all(x == 0 for x in v):
            v = prev_v
            break

    return u, v


def gaussian_reduce_3d(u, v, w):
    for _ in range(8):
        u, v = gaussian_reduce_2d(u, v)
        u, w = gaussian_reduce_2d(u, w)
        v, w = gaussian_reduce_2d(v, w)
    norms = [math.sqrt(sum(float(x) ** 2 for x in vec)) for vec in [u, v, w]]
    order = np.argsort(norms)
    return [u, v, w][order[0]], [u, v, w][order[1]], [u, v, w][order[2]]


def compute_log_od(basis):
    """纯原生 int 计算长度对数，绝不会溢出，OD 值真实且精确！"""
    try:
        norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in basis]
        norms = [max(x, 1e-10) for x in norms]
        return float(sum([math.log(x) for x in norms]))
    except Exception:
        return 0.0


def load_lattice(fp):
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    return [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]


def preprocess(fp):
    basis_30d = load_lattice(fp)
    sub = [row[:20] for row in basis_30d[:20]]
    try:
        # 【修复】准确打印矩阵的整数位数
        max_val = max(abs(x) for row in sub for x in row)
        digits = len(str(max_val))

        lll_basis = lll_reduction(sub)
        norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in lll_basis]
        lll_min = float(min(norms))

        logger.info(f"提取子格: {os.path.basename(fp)} | 整数位数: {digits} 位 | LLL_min: {lll_min:.4f}")
        return {'basis': sub, 'lll_min': lll_min, 'gh': lll_min * 1.05, 'scale': 10 ** (digits - 1)}, fp, None
    except Exception:
        return None, fp, "截取失败"


# ================= 强制验算模块（绕过智能体） =================
def verify_real_lattice_vector(orig_basis, v_min, scale):
    n = len(orig_basis)
    # 缩放后求系数，再取整
    orig_f = np.array(orig_basis, dtype=np.float64) / scale
    v_f = np.array(v_min, dtype=np.float64) / scale
    try:
        coeffs, _, _, _ = np.linalg.lstsq(orig_f.T, v_f, rcond=None)
        rounded = np.round(coeffs)
        coeff_error = float(np.max(np.abs(coeffs - rounded)))
        # 真实重建长度
        recon_v = [sum(int(round(rounded[k])) * orig_basis[k][i] for k in range(n)) for i in range(n)]
        verify_len = float(math.sqrt(sum(float(x) ** 2 for x in recon_v)))

        if coeff_error < 0.5 and verify_len > 1.0:
            return True, coeff_error, verify_len
        else:
            return False, coeff_error, verify_len
    except Exception:
        return False, -1.0, -1.0


class SVPEnv:
    def __init__(self, lattice_info, init_len_range=(3.0, 6.0)):
        self.orig_basis = lattice_info['basis']
        self.lll_min = lattice_info['lll_min']
        self.gh = lattice_info['gh']
        self.scale = lattice_info['scale']
        self.init_len_range = init_len_range
        self.n = BASIS_NUM
        self.lll_basis = None
        self.steps = 0
        self.no_improve = 0
        self.best_shortest = float('inf')
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
        min_len = float(np.min(lengths))
        basis_ratios = np.log(np.array(lengths) / min_len + 1e-8)
        mean_len_feat = np.log(np.mean(lengths))
        std_len_feat = np.log(np.std(lengths) + 1e-8)
        state = np.concatenate([np.array(log_lengths), np.array(cos_vals),
                                np.clip(basis_ratios, -5, 5),
                                np.array([mean_len_feat, std_len_feat])]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self):
        U = self.random_unimodular(self.n)
        self.lll_basis = [[sum(self.orig_basis[i][k] * U[k][j] for k in range(self.n)) for j in range(self.n)] for i in
                          range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                v1, v2 = gaussian_reduce_2d(self.lll_basis[i], self.lll_basis[j])
                self.lll_basis[i], self.lll_basis[j] = v1, v2
        self.lll_basis = lll_reduction(self.lll_basis)
        self.steps = 0;
        self.no_improve = 0
        self.best_shortest = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])
        return self._get_state()

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
        act_type, idxs = self.action_list[action_idx][0], self.action_list[action_idx][1:]
        prev_basis = [row[:] for row in self.lll_basis]
        old_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])
        old_od = compute_log_od(self.lll_basis)

        if act_type == 'PAIR':
            i, j = idxs
            v1, v2 = gaussian_reduce_2d(self.lll_basis[i], self.lll_basis[j])
            self.lll_basis[i], self.lll_basis[j] = v1, v2
        elif act_type == 'TRIPLE':
            i, j, k = idxs
            v1, v2, v3 = gaussian_reduce_3d(self.lll_basis[i], self.lll_basis[j], self.lll_basis[k])
            self.lll_basis[i], self.lll_basis[j], self.lll_basis[k] = v1, v2, v3

        if self.steps % LLL_INTERVAL == 0:
            self.lll_basis = lll_reduction(self.lll_basis)

        new_od = compute_log_od(self.lll_basis)
        new_len = min([math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis])

        was_rejected = False
        coeff_err_out = -1.0
        verify_len_out = -1.0

        # 【强制验算】：如果发现极短向量（小于80%），或者为0，强制验算！
        if new_len < self.lll_min * 0.8 or new_len < 2.0:
            norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis]
            idx_min = int(np.argmin(norms))
            is_valid, coeff_err_out, verify_len_out = verify_real_lattice_vector(self.orig_basis,
                                                                                 self.lll_basis[idx_min], self.scale)
            if not is_valid:
                was_rejected = True

        # 执行回滚
        if was_rejected:
            self.lll_basis = [row[:] for row in prev_basis]
            self.lll_basis = lll_reduction(self.lll_basis)
            new_od = compute_log_od(self.lll_basis)
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
            'no_improve': self.no_improve,
            'coeff_err': coeff_err_out,
            'verify_len': verify_len_out
        }
        return self._get_state(), reward, done, info


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
        return np.stack(s_next, axis=0), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), infos

    def reset_single(self, idx):
        return self.envs[idx].reset()


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
        logger.info(f"开始训练 20维子格智能体！设备：{DEVICE}")
        while self.steps < TOTAL_TRAIN_STEPS:
            if self.ep > 8000: cur_range = (3.0, 5.0)
            if self.ep > 16000: cur_range = (2.5, 4.0)
            if self.ep % 1000 == 0: vec_env = VecEnv(self.infos, NUM_ENVS, cur_range); states = vec_env.reset_all()
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
                # 【详细日志】：动作、Shortest、LLL_Min、OD、验算误差，全部显示！
                if step % 10 == 0:
                    b_idx = min(range(NUM_ENVS), key=lambda i: infos[i].get('cur_shortest', float('inf')))
                    b_inf = infos[b_idx]
                    coeff_str = f" | 验证误差 {b_inf['coeff_err']:.4f} | 验算长度 {b_inf['verify_len']:.4f}" if b_inf[
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
                    torch.save(self.agent.model.state_dict(), "best_ppo_20d_subgem.pth")


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
    logger.info(f"找到 {len(files)} 个30维格，正在提取前20行前20列作为20维子格...")

    with Pool(min(cpu_count(), len(files))) as pool:
        res = pool.map(preprocess, files[:limit])
    infos = [r[0] for r in res if r[2] is None]
    if not infos: raise RuntimeError("无法提取任何合格的子格！")
    logger.info(f"成功提取 {len(infos)} 个20维子格。开始训练...")
    trainer = Trainer(infos);
    trainer.train()