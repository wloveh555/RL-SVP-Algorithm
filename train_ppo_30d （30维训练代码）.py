"""
30维全量训练与深度突破版（500个矩阵，精简日志版，提升速度）
"""
import os, glob, math, random, logging, numpy as np, time
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
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

EPS = 1e-8
LATTICE_DIR = r"/mnt/d/generator/三十维"
FILE_PATTERN = "random_lattice_30_*"

BASIS_NUM = 30
USE_3D = True
ACTION_DIM = 4495 if USE_3D else 435
STATE_DIM = 467  # 30 + 435 + 2
PPO_EPOCHS = 10
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 1e-4
ENTROPY_COEF = 0.05
MAX_STEPS_EPISODE = 500
TOTAL_TRAIN_STEPS = 100000
NUM_ENVS = 16
LLL_INTERVAL = 100
NO_IMPROVE_STEP_LIMIT = 60
BATCH_SIZE = 1024
HIDDEN_DIM = 256
SAFE_MIN_THRESHOLD = 2.0
MODEL_PATH = "best_ppo_30d_full.pth"
LOG_FILE = 'ppo_training_log_30d_full.txt'

logging.basicConfig(filename=LOG_FILE, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO);
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


def compute_gh(logdet, n):
    log_gamma = math.lgamma(n / 2 + 1)
    gh_log = math.log(1.05) + (log_gamma / n) - 0.5 * math.log(math.pi) + (logdet / n)
    return math.exp(gh_log)


def lll_reduction(basis, delta=0.99):
    n = len(basis)
    int_basis_list = [[int(x) for x in row] for row in basis]
    B_mat = IntegerMatrix.from_matrix(int_basis_list)
    result = LLL.reduction(B_mat, delta=delta)
    B_out = result[0] if isinstance(result, tuple) else result
    return [[int(row[j]) for j in range(n)] for row in B_out]


def gaussian_reduce_2d(u, v, max_iter=50):
    u_f = np.array(u, dtype=np.float64);
    v_f = np.array(v, dtype=np.float64)
    it = 0
    while it < max_iter:
        it += 1
        norm_u_f = np.dot(u_f, u_f);
        norm_v_f = np.dot(v_f, v_f)
        if norm_u_f > norm_v_f: u, v = v, u; u_f, v_f = v_f, u_f; norm_u_f, norm_v_f = norm_v_f, norm_u_f
        mu_f = np.dot(v_f, u_f) / max(norm_u_f, 1e-300)
        if not np.isfinite(mu_f) or round(mu_f) == 0: break
        m = int(round(mu_f))
        v = [v[i] - m * u[i] for i in range(len(u))]
        v_f = np.array(v, dtype=np.float64)
    return u, v


def gaussian_reduce_3d(u, v, w):
    for _ in range(8):
        u, v = gaussian_reduce_2d(u, v);
        u, w = gaussian_reduce_2d(u, w);
        v, w = gaussian_reduce_2d(v, w)
    norms = [np.linalg.norm(np.array(x, dtype=np.float64)) for x in [u, v, w]]
    order = np.argsort(norms)
    return [u, v, w][order[0]], [u, v, w][order[1]], [u, v, w][order[2]]


def compute_ortho_def(basis, logdet):
    try:
        norms = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in basis]
        norms = [max(x, 1e-300) for x in norms]
        log_od = sum(math.log(x) for x in norms) - logdet
        return math.exp(log_od) if log_od <= 700 else float('inf')
    except Exception:
        return float('inf')


def load_lattice(fp):
    with open(fp, 'r') as f:
        content = f.read().strip().replace('[', ' ').replace(']', ' ').replace('\n', ' ')
    nums = content.split()
    n = int(round(math.sqrt(len(nums))))
    basis = [[int(nums[i * n + j]) for j in range(n)] for i in range(n)]
    return basis


def random_unimodular(n, max_val=5):
    L = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j: L[i][j] = 1
    for i in range(1, n):
        for j in range(i): L[i][j] = random.randint(-max_val, max_val + 1)
    return L


def preprocess(fp):
    basis = load_lattice(fp)
    try:
        basis_f = np.array(basis, dtype=np.float64)
        sign, logdet = np.linalg.slogdet(basis_f)
        if sign == 0: return None, fp, "Singular"
    except Exception as e:
        return None, fp, f"行列式计算失败: {str(e)}"
    lll_basis = lll_reduction(basis)
    lll_min = min(math.sqrt(sum(float(x) ** 2 for x in row)) for row in lll_basis)
    gh = compute_gh(float(logdet), BASIS_NUM)
    logger.info(f"预处理: {os.path.basename(fp)} | LLL_min: {lll_min:.4f} | 估算GH: {gh:.4f}")
    return {'basis': basis, 'logdet': float(logdet), 'lll_min': float(lll_min), 'gh': float(gh)}, fp, None


class SVPEnv:
    def __init__(self, lattice_info):
        self.orig_basis = lattice_info['basis']
        self.logdet = lattice_info['logdet']
        self.lll_min = lattice_info['lll_min']
        self.gh = lattice_info['gh']
        self.n = BASIS_NUM
        self.lll_basis = None
        self.steps = 0
        self.no_improve = 0
        self.best_shortest = float('inf')
        self.best_od = float('inf')
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
        lengths = [math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis]
        log_lengths = [math.log(max(l, 1e-300)) for l in lengths]
        cos_vals = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                denom = lengths[i] * lengths[j]
                if denom == 0:
                    cos_vals.append(0.0)
                else:
                    num = sum(float(self.lll_basis[i][k]) * float(self.lll_basis[j][k]) for k in range(self.n))
                    cos_vals.append(abs(num) / denom)
        shortest = min(lengths)
        feat_1 = math.log(shortest / max(self.lll_min, 1e-300))
        feat_2 = math.log(shortest / max(self.gh, 1e-300))
        state = np.concatenate([np.array(log_lengths), np.array(cos_vals), np.array([feat_1, feat_2])]).astype(
            np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)

    def reset(self):
        U = random_unimodular(self.n)
        self.lll_basis = [[sum(self.orig_basis[i][k] * U[k][j] for k in range(self.n)) for j in range(self.n)] for i in
                          range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                v1, v2 = gaussian_reduce_2d(self.lll_basis[i], self.lll_basis[j])
                self.lll_basis[i], self.lll_basis[j] = v1, v2
        self.lll_basis = lll_reduction(self.lll_basis)
        self.steps = 0;
        self.no_improve = 0
        self.best_shortest = min(math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis)
        self.best_od = compute_ortho_def(self.lll_basis, self.logdet)
        return self._get_state()

    def step(self, action_idx):
        self.steps += 1
        act_type, idxs = self.action_list[action_idx][0], self.action_list[action_idx][1:]
        prev_basis = [row[:] for row in self.lll_basis]
        old_od = compute_ortho_def(self.lll_basis, self.logdet)
        old_len = min(math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis)

        # 【修改】已删除原先每一步的 logger.info("选择动作...") 以提升速度并精简日志

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

        new_od = compute_ortho_def(self.lll_basis, self.logdet)
        new_len = min(math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis)

        was_rejected = False
        is_nan = not np.isfinite(new_len) or not np.isfinite(new_od)
        is_fake_zero = new_len <= SAFE_MIN_THRESHOLD
        is_exploded = new_od > 1e50 or new_len > self.lll_min * 5

        if is_nan or is_fake_zero or is_exploded:
            self.lll_basis = [row[:] for row in prev_basis]
            self.lll_basis = lll_reduction(self.lll_basis)
            new_od = compute_ortho_def(self.lll_basis, self.logdet)
            new_len = min(math.sqrt(sum(float(x) ** 2 for x in row)) for row in self.lll_basis)
            was_rejected = True

        if new_od < self.best_od:
            self.best_od = new_od; self.no_improve = 0
        elif new_len < self.best_shortest:
            self.best_shortest = new_len; self.no_improve = 0
        else:
            self.no_improve += 1

        reward = 0.0
        if was_rejected:
            reward = -10.0
        else:
            reward = (math.log(old_od) - math.log(new_od)) * 20.0 + (math.log(old_len) - math.log(new_len)) * 50.0
            if new_len < self.lll_min * 0.8: reward += 200.0
            if new_len < self.lll_min * 0.5: reward += 1000.0
            if new_len < 500.0: reward += 2000.0
            if new_len < 100.0: reward += 5000.0
            if self.no_improve > NO_IMPROVE_STEP_LIMIT: reward -= 10.0
            reward = np.clip(reward, -10.0, 5000.0)

        done = self.steps >= MAX_STEPS_EPISODE
        info = {
            'status': 'REJECTED' if was_rejected else 'RUNNING',
            'action': action_idx, 'pair': idxs,
            'cur_shortest': new_len, 'lll_min': self.lll_min,
            'new_od': new_od, 'no_improve': self.no_improve,
            'gh': self.gh
        }
        return self._get_state(), reward, done, info


class VecEnv:
    def __init__(self, lat_list, num_envs):
        self.envs = [SVPEnv(random.choice(lat_list)) for _ in range(num_envs)]

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


class Trainer:
    def __init__(self, infos):
        self.infos = infos;
        self.agent = PPOAgent()
        self.steps = 0;
        self.ep = 0;
        self.best_od = float('inf')
        self.start_time = time.time()

    def train(self):
        vec_env = VecEnv(self.infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"加载了 {len(self.infos)} 个格子，开始训练！设备：{DEVICE}")

        while self.steps < TOTAL_TRAIN_STEPS:
            buf = [[] for _ in range(6)];
            buf_infos = []
            for step in range(MAX_STEPS_EPISODE):
                actions, lp, v = self.agent.get_action(states)
                s_next, r, d, infos = vec_env.step_all(actions)

                for i in range(NUM_ENVS):
                    if infos[i].get('no_improve', 0) >= NO_IMPROVE_STEP_LIMIT:
                        d[i] = True;
                        r[i] -= 10
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

                # 【修改】调整为每 200 步打印一次，大幅精简日志
                if step % 200 == 0:
                    b_idx = min(range(NUM_ENVS), key=lambda i: infos[i].get('cur_shortest', float('inf')))
                    b_inf = infos[b_idx]
                    elapsed = time.time() - self.start_time
                    speed = self.steps / elapsed
                    logger.info(
                        f"Step {self.steps} | 速度: {speed:.2f} 步/秒 | 当前最优动作: {b_inf.get('pair', '')} | Best Shortest {b_inf['cur_shortest']:.4f} | LLL_Min {b_inf['lll_min']:.4f} | Status: {b_inf['status']}")

            _, _, last_v = self.agent.get_action(states)
            buf[5].append(last_v)
            T = MAX_STEPS_EPISODE
            all_s = np.stack(buf[0], 0).transpose(1, 0, 2).reshape(-1, STATE_DIM)
            all_a = np.stack(buf[1], 0).transpose(1, 0).reshape(-1)
            all_lp = np.stack(buf[2], 0).transpose(1, 0).reshape(-1)
            all_r = np.stack(buf[3], 0).transpose(1, 0).reshape(-1)
            all_d = np.stack(buf[4], 0).transpose(1, 0).reshape(-1)
            all_v = np.stack(buf[5][:-1], 0).transpose(1, 0).reshape(-1)

            all_r_mean = np.mean(all_r);
            all_r_std = np.std(all_r)
            all_r_norm = (all_r - all_r_mean) / (all_r_std + 1e-8)
            all_adv, all_ret = np.zeros_like(all_r_norm), np.zeros_like(all_r_norm)

            for ei in range(NUM_ENVS):
                st = ei * T;
                ga = np.zeros(T, dtype=np.float32);
                last_gae = 0
                for t in reversed(range(T)):
                    mask = 1 - all_d[st + t]
                    nv = 0 if t == T - 1 else all_v[st + t + 1]
                    delta = all_r_norm[st + t] + GAMMA * nv * mask - all_v[st + t]
                    last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
                    ga[t] = last_gae
                all_adv[st:st + T] = ga;
                all_ret[st:st + T] = ga + all_v[st:st + T]

            loss = self.agent.update(all_s, all_a, all_lp, all_adv, all_ret)

            avg_no_improve = np.mean([info.get('no_improve', 0) for info in infos])
            if avg_no_improve > 40:
                self.agent.entropy = min(0.3, self.agent.entropy * 1.2)
                logger.info(f"【智能体总结】发现长期停滞，强制提升探索熵至: {self.agent.entropy:.3f}")

            self.ep += NUM_ENVS
            info_0 = buf_infos[-1][0]
            if info_0.get('new_od', float('inf')) < self.best_od:
                self.best_od = info_0.get('new_od')
                torch.save(self.agent.model.state_dict(), MODEL_PATH)
                logger.info(f"Ep {self.ep} | Loss {loss:.4f} | 已保存新最佳模型: {MODEL_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lattices", type=int, default=0, help="0表示加载全部")
    args = parser.parse_args()

    files = glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN))
    if not files: raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}")
    limit = len(files) if args.max_lattices <= 0 else min(args.max_lattices, len(files))
    logger.info(f"发现 {len(files)} 个矩阵，将加载前 {limit} 个用于训练...")
    logger.info("注意：由于全部使用的是 10^90 级别纯 Python 大整数，加载全部500个将显著增加预处理时间，请耐心等待...")

    with Pool(min(cpu_count(), len(files[:limit]))) as pool:
        res = pool.map(preprocess, files[:limit])

    infos = [r[0] for r in res if r[2] is None]
    if not infos: raise RuntimeError("无可用格文件！")

    logger.info(f"成功加载 {len(infos)} 个格子。开始突破性训练...")
    trainer = Trainer(infos);
    trainer.train()