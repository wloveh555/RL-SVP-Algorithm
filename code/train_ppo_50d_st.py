"""
50维 Set Transformer 训练代码 V50-ST (路线 B)

核心改进：
  ★ 状态不再展平，而是构建为 (50, 100) 的矩阵（每行包含自身长度、与其他行的 cos、与其他行的 GSO-μ）
  ★ 网络改用 Set Transformer (Attention) 处理行间特征
  ★ 隐式地融合了 GSO 系数，避免之前直接拼接 780 维导致网络崩塌的问题

环境设置保持 V50-CONTIG.2 不变（连续窗口 K=35，LLL起点0.7，兜底100）。

命名：
  代码：train_ppo_50d_st.py
  模型：best_ppo_50d_st.pth
  日志：ppo_training_log_50d_st.txt
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

# ===================== 50维配置 (ST版) =====================
BASIS_NUM_50 = 50
LATTICE_DIR = "../demo" 
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
BATCH_SIZE = 128  # ★ Transformer 显存占用大，Batch 减半
EMBED_DIM = 128  # ★ Transformer 隐藏层维度
NUM_HEADS = 4  # ★ 多头注意力
NUM_LAYERS = 2  # ★ 堆叠层数
K_USE_BKZ = True

GROUND_TRUTH_PKL = "svp_ground_truth_50.pkl"
MODEL_SAVE_PATH = "best_ppo_50d_st.pth"
LOG_FILE = "ppo_training_log_50d_st.txt"

# ★ 输入维度重构：每行 1(loglen) + 49(cos) + 50(mu) = 100 维
ROW_FEAT_DIM = 100
# 环境返回给 Agent 的扁平维度：50 * 100 + 2 (全局特征) = 5002
STATE_DIM_FLAT = BASIS_NUM_50 * ROW_FEAT_DIM + 2

logging.basicConfig(filename=LOG_FILE, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
console = logging.StreamHandler();
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger()


# ===================== 工具函数 =====================
def norm_sq(row): return sum(int(x) * int(x) for x in row)


def min_norm_sq(basis): return min(norm_sq(row) for row in basis)


def has_zero_row(basis): return any(all(int(x) == 0 for x in row) for row in basis)


def sqrt_float_from_sq(nsq): return 0.0 if nsq <= 0 else math.exp(0.5 * math.log(nsq))


def load_lattice_50d(file_name, lattice_dir=None):
    if lattice_dir is None: lattice_dir = LATTICE_DIR
    fp = os.path.join(lattice_dir, file_name)
    if not os.path.exists(fp):
        for ext in ['.txt', '']:
            cand = os.path.join(lattice_dir, file_name + ext)
            if os.path.exists(cand): fp = cand; break
        else:
            base = file_name.replace('.txt', '')
            matched = [f for f in glob.glob(os.path.join(lattice_dir, "*")) if
                       os.path.basename(f).replace('.txt', '') == base]
            if not matched: raise FileNotFoundError(f"未找到 {file_name}")
            fp = matched[0]
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
        if K_USE_BKZ:
            par = BKZ.Param(block_size=k, strategies=BKZ.DEFAULT_STRATEGY, max_loops=1)
            BKZ.reduction(B, par)
        else:
            LLL.reduction(B)
            try:
                v = SVP.shortest_vector(B)
                v_int = [int(x) for x in v]
                if len(v_int) != BASIS_NUM_50 or all(x == 0 for x in v_int): return None, -1
                v_sq = sum(x * x for x in v_int)
                return (v_int, v_sq) if v_sq > 0 else (None, -1)
            except Exception:
                return None, -1
        best_sq, best_v = float('inf'), None
        for i in range(k):
            row = [int(B[i, j]) for j in range(BASIS_NUM_50)]
            sq = sum(x * x for x in row)
            if 0 < sq < best_sq: best_sq, best_v = sq, row
        return (best_v, best_sq) if best_v else (None, -1)
    except Exception:
        return None, -1


# ===================== ★ 核心：构造 (50, 100) 行特征矩阵 =====================
def compute_state_matrix(basis, episode_start_sq, step, max_steps):
    n = BASIS_NUM_50
    norm_sqs = [max(norm_sq(row), 1) for row in basis]
    log_lengths = [0.5 * math.log(nsq) for nsq in norm_sqs]

    # 计算 GSO mu
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

    # 构造特征矩阵
    feat = np.zeros((n, ROW_FEAT_DIM), dtype=np.float32)
    for i in range(n):
        feat[i, 0] = log_lengths[i]  # 第0维：自身长度
        idx = 1
        for j in range(n):
            if i == j: continue
            dot = sum(int(basis[i][k]) * int(basis[j][k]) for k in range(n))
            cos_val = 0.0
            if dot != 0:
                log_cos_sq = 2 * math.log(abs(dot)) - math.log(norm_sqs[i]) - math.log(norm_sqs[j])
                cos_val = min(abs(math.exp(0.5 * min(0.0, log_cos_sq))), 1.0)
            feat[i, idx] = cos_val
            # 后面 50 维放 mu（保持对称性，方便 Attention 学习）
            feat[i, idx + 49] = mu[i][j] if i > j else mu[j][i]
            idx += 1

    # 全局标量
    cur_min = min_norm_sq(basis)
    start_ratio = math.log(max(episode_start_sq, 1)) - math.log(
        max(cur_min, 1)) if episode_start_sq > 0 and cur_min > 0 else 0.0
    global_feat = np.array([start_ratio, step / max(max_steps, 1)], dtype=np.float32)

    feat = np.clip(np.nan_to_num(feat, nan=0.0, posinf=5.0, neginf=-5.0), -5.0, 5.0)
    return feat, global_feat


# ===================== 环境 =====================
def _worker_step(args):
    basis, selected_rows = args
    prev_basis = [r[:] for r in basis]
    old_min_sq = min_norm_sq(basis)
    zero_incidents = 0
    v, v_sq = enumerate_sub_lattice(basis, list(selected_rows))
    if v is not None and v_sq > 0 and v_sq < old_min_sq and any(x != 0 for x in v):
        basis[int(np.argmax([norm_sq(r) for r in basis]))] = v
        if has_zero_row(basis): basis = prev_basis; zero_incidents += 1
    if has_zero_row(basis): basis = prev_basis; zero_incidents += 1
    return basis, zero_incidents


class SVPEnv50:
    def __init__(self, lat_list):
        if not isinstance(lat_list, list): lat_list = [lat_list]
        self.lat_list = lat_list
        self._bind_lattice(random.choice(lat_list))
        self.basis, self.steps, self.no_improve = None, 0, 0
        self.best_shortest_sq, self.episode_start_sq, self.zero_row_incidents = 0, 0, 0

    def _bind_lattice(self, info):
        self.orig_basis = info['basis'] if 'basis' in info and info['basis'] else load_lattice_50d(info['file'])
        self.lll_min_sq = int(info['lll_min_sq'])
        self.bkz30_min_sq = int(info.get('bkz30_min_sq', info['lll_min_sq']))
        self.svp_min_sq = int(info.get('svp_min_sq', info['lll_min_sq']))
        self.gh = float(info.get('gh', 0.0))

    def reset(self, fixed_lattice=None):
        if fixed_lattice is not None:
            self._bind_lattice(fixed_lattice)
        else:
            self._bind_lattice(random.choice(self.lat_list))
        self.basis = lll_reduction_safe(self.orig_basis, delta=START_LLL_DELTA)
        n = BASIS_NUM_50
        U = [[0] * n for _ in range(n)]
        for i in range(n): U[i][i] = 1
        for i in range(1, n):
            for j in range(i): U[i][j] = random.randint(-RESET_UNIMODULAR_MAX_VAL, RESET_UNIMODULAR_MAX_VAL + 1)
        new_basis = [[sum(int(U[i][k]) * int(self.basis[k][j]) for k in range(n)) for j in range(n)] for i in range(n)]
        if not has_zero_row(new_basis): self.basis = new_basis
        self.steps, self.no_improve = 0, 0
        self.best_shortest_sq = min_norm_sq(self.basis)
        self.episode_start_sq = self.best_shortest_sq
        self.zero_row_incidents = 0
        # 返回展平后的数组，方便 multiprocessing
        feat, glob = compute_state_matrix(self.basis, self.episode_start_sq, 0, MAX_STEPS_EPISODE)
        return np.concatenate([feat.flatten(), glob])

    def apply_step_result(self, new_basis, zero_incidents):
        self.basis = new_basis
        self.zero_row_incidents += zero_incidents
        self.steps += 1
        if LLL_FALLBACK_INTERVAL > 0 and self.steps % LLL_FALLBACK_INTERVAL == 0:
            new_b = lll_reduction_safe(self.basis, delta=0.99)
            if not has_zero_row(new_b): self.basis = new_b
        new_min_sq = min_norm_sq(self.basis)
        if new_min_sq < self.best_shortest_sq:
            self.best_shortest_sq = new_min_sq
            self.no_improve = 0
        else:
            self.no_improve += 1


class PoolEnv50:
    def __init__(self, lat_list, num_envs):
        self.envs = [SVPEnv50(lat_list) for _ in range(num_envs)]
        self.pool = Pool(processes=min(num_envs, cpu_count()))

    def reset_all(self):
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step_all(self, selections):
        results = self.pool.map(_worker_step, [(env.basis, sel) for env, sel in zip(self.envs, selections)])
        s_next, rewards, dones, infos = [], [], [], []
        for i, (env, (new_basis, zero_inc)) in enumerate(zip(self.envs, results)):
            old_min_sq = min_norm_sq(env.basis)
            env.apply_step_result(new_basis, zero_inc)
            new_min_sq = min_norm_sq(env.basis)
            start_len, old_len, new_len = sqrt_float_from_sq(env.episode_start_sq), sqrt_float_from_sq(
                old_min_sq), sqrt_float_from_sq(new_min_sq)

            reward = 0.0
            if start_len > 1e-8 and new_len > 1e-8:
                sg = math.log(start_len) - math.log(new_len)
                if sg > 0: reward += sg * REWARD_START_GAIN
            if old_len > 1e-8 and new_len > 1e-8:
                sg = math.log(old_len) - math.log(new_len)
                if sg > 0: reward += sg * REWARD_STEP_GAIN
            if new_min_sq >= old_min_sq: reward += REWARD_INVALID
            if env.no_improve > NO_IMPROVE_STEP_LIMIT: reward -= 5.0

            done = env.steps >= MAX_STEPS_EPISODE

            feat, glob = compute_state_matrix(env.basis, env.episode_start_sq, env.steps, MAX_STEPS_EPISODE)
            s_next.append(np.concatenate([feat.flatten(), glob]))
            rewards.append(float(np.clip(reward, -10.0, 500.0)))
            dones.append(done)
            infos.append({'cur_shortest': new_len, 'start_shortest_sq': env.episode_start_sq,
                          'lll_min': sqrt_float_from_sq(env.lll_min_sq),
                          'bkz30_min': sqrt_float_from_sq(env.bkz30_min_sq),
                          'svp_min': sqrt_float_from_sq(env.svp_min_sq), 'zero_incidents': env.zero_row_incidents,
                          'no_improve': env.no_improve})

        return np.stack(s_next, axis=0), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.float32), infos

    def reset_single(self, idx):
        return self.envs[idx].reset()

    def close(self):
        self.pool.close(); self.pool.join()


# ===================== ★ 核心网络：Set Transformer =====================
class ActorCritic50_ST(nn.Module):
    def __init__(self):
        super().__init__()
        # 把 100 维行特征投影到 128 维
        self.row_proj = nn.Sequential(nn.Linear(ROW_FEAT_DIM, EMBED_DIM), nn.LayerNorm(EMBED_DIM), nn.ReLU())

        # 2 层 Transformer，4 个注意力头
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=EMBED_DIM * 2,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        # 输出每行被选中为起点的概率
        self.selector = nn.Linear(EMBED_DIM, 1)

        # Critic 综合所有行特征 + 全局特征
        self.critic = nn.Sequential(nn.Linear(EMBED_DIM + 2, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        B = x.shape[0]
        # 拆解输入，前 50*100 重新 view 成 (B, 50, 100)
        row_feat = x[:, :BASIS_NUM_50 * ROW_FEAT_DIM].view(B, BASIS_NUM_50, ROW_FEAT_DIM)
        global_feat = x[:, -2:]

        h = self.row_proj(row_feat)  # (B, 50, 128)
        h = self.transformer(h)  # (B, 50, 128)

        rows_logits = self.selector(h).squeeze(-1)  # (B, 50)

        h_pool = h.mean(dim=1)  # 全局平均池化 (B, 128)
        value = self.critic(torch.cat([h_pool, global_feat], dim=-1)).squeeze(-1)  # (B,)

        return torch.nan_to_num(rows_logits, nan=0.0), torch.nan_to_num(value, nan=0.0)


class PPOAgent50_ST:
    def __init__(self):
        self.model = ActorCritic50_ST().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR_MAX)

    def get_action(self, states, det=False):
        s = torch.from_numpy(states).to(DEVICE)
        with torch.no_grad():
            rows_logits, values = self.model(s)
            if det:
                start_idx = rows_logits.argmax(dim=-1)
            else:
                start_idx = Categorical(logits=rows_logits).sample()

            arange_k = torch.arange(K_SELECT, device=DEVICE).unsqueeze(0)
            selected = (start_idx.unsqueeze(1) + arange_k) % BASIS_NUM_50

            log_probs = F.log_softmax(rows_logits, dim=-1).gather(1, start_idx.unsqueeze(-1)).squeeze(-1)
        return selected.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def update(self, states, selections, old_lp, adv, ret, entropy_coef):
        self.model.train()
        s = torch.from_numpy(states).to(DEVICE)
        sel, old, adv, ret = torch.LongTensor(selections).to(DEVICE), torch.FloatTensor(old_lp).to(
            DEVICE), torch.FloatTensor(adv).to(DEVICE), torch.FloatTensor(ret).to(DEVICE)

        adv_mean, adv_std = adv.mean(), adv.std()
        adv = (adv - adv_mean) / (adv_std + EPS) if adv_std.item() > 1e-6 else adv - adv_mean
        old, adv, ret = torch.nan_to_num(old, nan=0.0), torch.nan_to_num(adv, nan=0.0), torch.nan_to_num(ret, nan=0.0)

        total_loss, num, skipped = 0, 0, 0
        for _ in range(PPO_EPOCHS):
            idxs = np.random.permutation(len(s))
            for st in range(0, len(s), BATCH_SIZE):
                idx = idxs[st:st + BATCH_SIZE]
                rows_logits, values = self.model(s[idx])
                start_idx = sel[idx][:, 0]

                new_logp = F.log_softmax(rows_logits, dim=-1).gather(1, start_idx.unsqueeze(-1)).squeeze(-1)
                new_logp = torch.nan_to_num(new_logp, nan=0.0)

                p = F.softmax(rows_logits, dim=-1)
                ent = torch.nan_to_num(-(p * (p + 1e-10).log()).sum(dim=-1).mean(), nan=0.0)

                ratio = torch.exp(torch.clamp(new_logp - old[idx], -10.0, 10.0))
                surr1, surr2 = ratio * adv[idx], torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv[idx]
                loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(values, ret[idx]) - entropy_coef * ent

                if torch.isnan(loss) or torch.isinf(loss): skipped += 1; continue

                self.optimizer.zero_grad();
                loss.backward()
                if any(p_.grad is not None and (torch.isnan(p_.grad).any() or torch.isinf(p_.grad).any()) for p_ in
                       self.model.parameters()):
                    skipped += 1;
                    self.optimizer.zero_grad();
                    continue

                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item();
                num += 1

        if skipped > 0: logger.warning(f"  [PPO] 跳过 {skipped} 个 NaN batch")
        return total_loss / max(num, 1)


class RewardRM:
    def __init__(self):
        self.mean, self.var, self.count = 0, 1, 1e-4

    def update(self, x):
        x = np.clip(np.nan_to_num(x, nan=0), -1000, 1000)
        d = np.mean(x) - self.mean
        total = self.count + len(x)
        self.mean += d * len(x) / total
        self.var = max((self.var * self.count + np.var(x) * len(x) + d ** 2 * self.count * len(x) / total) / total, EPS)
        self.count = total

    def normalize(self, x):
        return (np.clip(np.nan_to_num(x, nan=0), -1000, 1000) - self.mean) / np.sqrt(self.var + EPS)


def get_lr(step, total):
    if step < WARMUP_STEPS: return LR_MAX * (step + 1) / WARMUP_STEPS
    if step < total * 0.8: return LR_MAX
    progress = (step - WARMUP_STEPS) / max(total * 0.2, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * progress))


# ===================== 训练器 =====================
class Trainer50_ST:
    def __init__(self, infos, val_infos):
        self.infos = infos
        self.val_infos = val_infos
        self.agent = PPOAgent50_ST()
        self.rms = RewardRM()
        self.steps = 0
        self.ep = 0
        self.total_zero_incidents = 0
        self.start_time = time.time()

    def train(self):
        vec_env = PoolEnv50(self.infos, NUM_ENVS)
        states = vec_env.reset_all()
        logger.info(f"开始训练 50维 V50-ST (Set Transformer)！设备：{DEVICE}")
        logger.info(f"K_SELECT={K_SELECT}  LLL_INTERVAL={LLL_FALLBACK_INTERVAL}  "
                    f"MAX_STEPS={MAX_STEPS_EPISODE}  START_LLL_DELTA={START_LLL_DELTA}")
        logger.info(f"网络参数: ROW_FEAT={ROW_FEAT_DIM}  EMBED={EMBED_DIM}  LAYERS={NUM_LAYERS}")
        logger.info(f"模型保存: {MODEL_SAVE_PATH}")

        try:
            while self.steps < TOTAL_TRAIN_STEPS:
                current_lr = get_lr(self.steps, TOTAL_TRAIN_STEPS)
                for pg in self.agent.optimizer.param_groups: pg['lr'] = current_lr

                progress = self.steps / TOTAL_TRAIN_STEPS
                entropy_coef = ENTROPY_COEF_HI * (1 - progress) + ENTROPY_COEF_LO * progress

                buf_s, buf_sel, buf_lp, buf_r, buf_d, buf_v, buf_i = [], [], [], [], [], [], []

                for step in range(MAX_STEPS_EPISODE):
                    selections, lp, v = self.agent.get_action(states)
                    current_states = states.copy()
                    s_next, r, d, infos = vec_env.step_all(selections)

                    for i in range(NUM_ENVS):
                        if d[i]: s_next[i] = vec_env.reset_single(i)

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
                        summary = " | ".join([f"E{i}:{infos[i]['cur_shortest']:.1f}" for i in range(NUM_ENVS)])
                        elapsed = time.time() - self.start_time
                        speed = self.steps / max(elapsed, 1)
                        starts = [int(sel[0]) for sel in selections]
                        logger.info(f"Step {self.steps} ({speed:.2f}/s) | lr={current_lr:.2e} | starts={starts}")
                        logger.info(f"  [{summary}]")
                        b_idx = min(range(NUM_ENVS), key=lambda i: infos[i].get('cur_shortest', float('inf')))
                        b = infos[b_idx]
                        logger.info(
                            f"  Best E{b_idx} | start={sqrt_float_from_sq(b['start_shortest_sq']):.1f} "
                            f"→ {b['cur_shortest']:.1f} | LLL={b['lll_min']:.1f} | BKZ50={b['bkz30_min']:.1f} | SVP={b['svp_min']:.1f}")

                _, _, last_v = self.agent.get_action(states)
                buf_v.append(last_v)

                T = MAX_STEPS_EPISODE
                all_s = np.stack(buf_s, 0).transpose(1, 0, 2).reshape(-1, STATE_DIM_FLAT)
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

                loss = self.agent.update(all_s, all_sel, all_lp, all_adv, all_ret, entropy_coef)
                self.ep += NUM_ENVS

                if self.ep % 200 < NUM_ENVS:
                    cur_zero = sum(inf.get('zero_incidents', 0) for inf in buf_i[-1])
                    self.total_zero_incidents += cur_zero
                    logger.info(f"Ep {self.ep} | Loss {loss:.4f} | zero(cum)={self.total_zero_incidents}")
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
    args = parser.parse_args()

    TOTAL_TRAIN_STEPS = args.steps
    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)

    if not os.path.exists(args.ground_truth):
        logger.warning(f"真值文件不存在，从原始格文件构建...")
        files = sorted(glob.glob(os.path.join(LATTICE_DIR, FILE_PATTERN)))
        if not files:
            raise FileNotFoundError(f"未找到格文件: {LATTICE_DIR}/{FILE_PATTERN}")
        if args.max_lattices > 0: files = files[:args.max_lattices]
        infos_all = []
        for fp in files:
            try:
                basis = load_lattice_50d(os.path.basename(fp))
                lll_basis = lll_reduction_safe(basis, delta=0.99)
                lll_min_sq = min_norm_sq(lll_basis)
                infos_all.append({
                    'file': os.path.basename(fp),
                    'basis': basis,
                    'lll_min_sq': int(lll_min_sq),
                    'bkz30_min_sq': int(lll_min_sq),
                    'svp_min_sq': int(lll_min_sq),
                    'gh': float('nan'),
                })
                logger.info(f"  {os.path.basename(fp)}: LLL(0.99)={sqrt_float_from_sq(lll_min_sq):.1f}")
            except Exception as e:
                logger.warning(f"  跳过 {fp}: {e}")
    else:
        with open(args.ground_truth, 'rb') as f:
            data = pickle.load(f)
        infos_all = data['lattices']
        if args.max_lattices > 0: infos_all = infos_all[:args.max_lattices]
        for info in infos_all:
            if 'basis' not in info or info['basis'] is None:
                try:
                    info['basis'] = load_lattice_50d(info['file'])
                except Exception:
                    info['basis'] = None
        infos_all = [i for i in infos_all if i['basis'] is not None]

    logger.info(f"总格数: {len(infos_all)}")

    infos = infos_all
    if len(infos) < 5:
        raise RuntimeError("⚠️ 格数量不足！请检查路径是否正确。")

    random.shuffle(infos)
    n_val = max(4, len(infos) // 5)
    val_infos = infos[:n_val]
    train_infos = infos[n_val:]
    logger.info(f"训练格: {len(train_infos)}  验证格: {len(val_infos)}")
    logger.info("开始训练...\n")

    trainer = Trainer50_ST(train_infos, val_infos)
    trainer.train()