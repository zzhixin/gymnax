#%%
import os
import sys

import time

import numpy as np
import jax

# jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium.envs.box2d.lunar_lander as gym_ll
from gymnasium.envs.box2d.lunar_lander import LunarLander as GymLunar
from gymnax.environments.box2d.lunar_lander import LunarLander as JaxLunar
from gymnax.environments.box2d.lunar_lander import EnvParams


def rollout_jax(env, key, state, actions, params):
    keys = jax.random.split(key, actions.shape[0] + 1)

    def step_fn(carry, inp):
        state = carry
        step_key, action = inp
        obs, next_state, reward, done, _ = env.step(step_key, state, action, params)
        return next_state, (obs, reward, done)

    return jax.lax.scan(step_fn, state, (keys[1:], actions))


rollout_jax_jit = jax.jit(rollout_jax, static_argnames=("env",))


def rollout_jax_batch(env, key, actions, params):
    steps = actions.shape[0]
    batch = actions.shape[1]
    keys = jax.random.split(key, (steps + 1) * batch).reshape(steps + 1, batch, 2)
    reset_keys = keys[0]
    step_keys = keys[1:]

    obs0, state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)

    def step_fn(carry, inputs):
        state = carry
        step_key, action = inputs
        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(step_key, state, action, params)
        return next_state, (obs, reward, done)

    return obs0, jax.lax.scan(step_fn, state, (step_keys, actions))


rollout_jax_batch_jit = jax.jit(rollout_jax_batch, static_argnames=("env",))


def run_gym(seed, actions, fixed_height=None):
    gym_ll.INITIAL_RANDOM = 0.0
    env = GymLunar(continuous=True, render_mode=None, enable_wind=False)
    obs, _ = env.reset(seed=seed)

    # If a fixed terrain is provided, force it by editing the generated height.
    if fixed_height is not None:
        W = gym_ll.VIEWPORT_W / gym_ll.SCALE
        H = gym_ll.VIEWPORT_H / gym_ll.SCALE
        chunks = len(fixed_height) - 1
        chunk_x = [W / (chunks - 1) * i for i in range(chunks)]
        helipad_y = H / 4
        env.helipad_y = helipad_y
        env.helipad_x1 = chunk_x[chunks // 2 - 1]
        env.helipad_x2 = chunk_x[chunks // 2 + 1]

        height = np.array(fixed_height, dtype=np.float32)
        height[chunks // 2 - 2 : chunks // 2 + 3] = helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(chunks)
        ]

        env.world.DestroyBody(env.moon)
        env.moon = env.world.CreateStaticBody(
            shapes=gym_ll.edgeShape(vertices=[(0.0, 0.0), (float(W), 0.0)])
        )
        env.sky_polys = []
        for i in range(chunks - 1):
            p1 = (float(chunk_x[i]), float(smooth_y[i]))
            p2 = (float(chunk_x[i + 1]), float(smooth_y[i + 1]))
            env.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            env.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

    traj = [obs.copy()]
    rewards = []
    for a in actions:
        obs, r, terminated, truncated, _ = env.step(a)
        traj.append(obs.copy())
        rewards.append(r)
        if terminated or truncated:
            break
    env.close()
    return np.array(traj), np.array(rewards)


#%%
def run_gymnax(env, seed, actions, fixed_height=None):
    key = jax.random.PRNGKey(seed)
    key_reset, key_rollout = jax.random.split(key)

    params = EnvParams(
        enable_wind=False,
        initial_random_force_scale=0.0,
        use_fixed_terrain=fixed_height is not None,
        fixed_height=jnp.array(fixed_height) if fixed_height is not None else EnvParams().fixed_height,
    )

    obs0, state = env.reset(key_reset, params)
    actions_jnp = jnp.asarray(actions)
    state, (obs_seq, rewards, dones) = rollout_jax_jit(
        env, key_rollout, state, actions_jnp, params
    )
    obs_seq, rewards, dones = jax.tree.map(
        jax.block_until_ready, (obs_seq, rewards, dones)
    )

    obs_np = np.array(obs_seq)
    rewards_np = np.array(rewards)
    dones_np = np.array(dones)

    if np.any(dones_np):
        done_idx = int(np.argmax(dones_np))
        obs_np = obs_np[: done_idx + 1]
        rewards_np = rewards_np[: done_idx + 1]

    traj = np.vstack([np.array(obs0)[None, :], obs_np])
    return traj, rewards_np


#%%
seed = 0
rng = np.random.RandomState(0)
actions = rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)

# Flat terrain at helipad height for deterministic comparison.
H = 400 / 30.0
helipad_y = H / 4.0

def make_flat_height(offset=0.0):
    return np.ones(12, dtype=np.float32) * (helipad_y + offset)

def make_slope_height():
    return np.linspace(helipad_y - 0.5, helipad_y + 0.5, 12, dtype=np.float32)

cases = [
    ("flat", make_flat_height(), 0),
    ("flat_low", make_flat_height(-0.2), 1),
    ("flat_high", make_flat_height(0.2), 2),
    ("slope", make_slope_height(), 3),
]

env_jax = JaxLunar(continuous=True)
print("jax devices:", jax.devices())

results = []
warmup_done = False
for name, fixed_height, case_seed in cases:
    case_rng = np.random.RandomState(case_seed)
    case_actions = case_rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
    t0 = time.perf_counter()
    traj_gym, rew_gym = run_gym(case_seed, case_actions, fixed_height=fixed_height)
    t1 = time.perf_counter()
    traj_jax, rew_jax = run_gymnax(
        env_jax, case_seed, case_actions, fixed_height=fixed_height
    )
    t2 = time.perf_counter()
    min_len = min(len(traj_gym), len(traj_jax))
    min_rew_len = min(len(rew_gym), len(rew_jax))

    obs_a = traj_gym[:min_len]
    obs_b = traj_jax[:min_len]
    obs_min = np.minimum(obs_a.min(axis=0), obs_b.min(axis=0))
    obs_max = np.maximum(obs_a.max(axis=0), obs_b.max(axis=0))
    obs_range = np.maximum(obs_max - obs_min, 1e-6)
    obs_diff = np.mean(np.abs(obs_a - obs_b) / obs_range)

    rew_a = rew_gym[:min_rew_len]
    rew_b = rew_jax[:min_rew_len]
    rew_min = min(rew_a.min(), rew_b.min())
    rew_max = max(rew_a.max(), rew_b.max())
    rew_range = max(rew_max - rew_min, 1e-6)
    rew_diff = np.mean(np.abs(rew_a - rew_b) / rew_range)
    if not warmup_done:
        t_jax = float("nan")
        warmup_done = True
    else:
        t_jax = t2 - t1
    results.append(
        (
            name,
            traj_gym.shape,
            traj_jax.shape,
            obs_diff,
            rew_diff,
            t1 - t0,
            t_jax,
        )
    )

for name, shape_g, shape_j, obs_diff, rew_diff, t_gym, t_jax in results:
    t_jax_str = "warmup" if np.isnan(t_jax) else f"{t_jax:.3f}s"
    print(
        f"{name:10s} traj {shape_g} vs {shape_j} | obs diff {obs_diff:.6f} | reward diff {rew_diff:.6f} | gym {t_gym:.3f}s | jax {t_jax_str}"
    )

fixed_height, plot_seed = cases[0][1], cases[0][2]
plot_rng = np.random.RandomState(plot_seed)
plot_actions = plot_rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
traj_gym, rew_gym = run_gym(plot_seed, plot_actions, fixed_height=fixed_height)
traj_jax, rew_jax = run_gymnax(env_jax, plot_seed, plot_actions, fixed_height=fixed_height)

#%%
# benchmark: vmap on gymnax lunarlander (cuda) vs loop over gym lunarlander
bench_batch = int(os.environ.get("BENCH_BATCH", "4096"))
bench_steps = int(os.environ.get("BENCH_STEPS", "200"))
bench_seed = int(os.environ.get("BENCH_SEED", "0"))
bench_warmup = os.environ.get("BENCH_WARMUP", "1") == "1"
bench_key = jax.random.PRNGKey(bench_seed)

bench_actions = np.random.RandomState(bench_seed).uniform(
    -1, 1, size=(bench_steps, bench_batch, 2)
).astype(np.float32)

bench_actions_jnp = jnp.asarray(bench_actions)
bench_params = EnvParams(
    enable_wind=False,
    initial_random_force_scale=0.0,
    use_fixed_terrain=True,
    fixed_height=make_flat_height(),
)

t0 = time.perf_counter()
envs = [GymLunar(continuous=True, render_mode=None, enable_wind=False) for _ in range(bench_batch)]
for i, env in enumerate(envs):
    env.reset(seed=bench_seed + i)
for t in range(bench_steps):
    for i, env in enumerate(envs):
        env.step(bench_actions[t, i])
for env in envs:
    env.close()
t1 = time.perf_counter()

env_jax_bench = JaxLunar(continuous=True)
if bench_warmup:
    obs0, (state, (obs_seq, rewards, dones)) = rollout_jax_batch_jit(
        env_jax_bench, bench_key, bench_actions_jnp, bench_params
    )
    jax.tree.map(jax.block_until_ready, (obs0, obs_seq, rewards, dones))

t2 = time.perf_counter()
obs0, (state, (obs_seq, rewards, dones)) = rollout_jax_batch_jit(
    env_jax_bench, bench_key, bench_actions_jnp, bench_params
)
jax.tree.map(jax.block_until_ready, (obs0, obs_seq, rewards, dones))
t3 = time.perf_counter()

warmup_str = "on" if bench_warmup else "off"
print(
    f"bench batch={bench_batch} steps={bench_steps} warmup={warmup_str} | gym {t1 - t0:.3f}s | jax {t3 - t2:.3f}s"
)

#%%
import matplotlib

if os.environ.get("COMPARE_PLOT", "1") == "0":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

t = np.arange(len(traj_gym))
labels = ["x", "y", "vx", "vy", "angle", "ang_v", "leg_l", "leg_r"]

fig, axs = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
axs = axs.ravel()
for i in range(8):
    axs[i].plot(t, traj_gym[:, i], label="gym", linewidth=1.5)
    axs[i].plot(
        t[: len(traj_jax)],
        traj_jax[:, i],
        label="jax",
        linewidth=1.0,
        linestyle="--",
    )
    axs[i].set_title(labels[i])
axs[0].legend()
plt.tight_layout()
if os.environ.get("COMPARE_PLOT", "1") == "1":
    plt.show()
else:
    plt.savefig(os.path.join(os.path.dirname(__file__), "compare_lunarlander_obs.png"), dpi=150)

plt.figure(figsize=(8, 3))
plt.plot(rew_gym, label="gym")
plt.plot(rew_jax, label="jax", linestyle="--")
plt.legend()
plt.title("Reward per step")
plt.tight_layout()
if os.environ.get("COMPARE_PLOT", "1") == "1":
    plt.show()
else:
    plt.savefig(os.path.join(os.path.dirname(__file__), "compare_lunarlander_reward.png"), dpi=150)
