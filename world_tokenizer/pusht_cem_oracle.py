"""Oracle-dynamics CEM diagnostic (PROGRESS.md pt. 23).

Same CEM knobs and the same registered block-pose cost as pusht_cem.py, but the
"dynamics model" is the TRUE simulator: candidate action sequences are rolled in
a scratch PushTEnv restored to the live env's exact body state (positions AND
velocities), stepping the verbatim PD + physics loop without the shapely reward.
Planner reads privileged true block pose for the cost. This upper-bounds what
the planning harness (cost + horizon + CEM) can achieve with a perfect model —
the fork that decides whether rung-0's nulls indict the learned dynamics or the
harness itself.

Protocol: PushTEnv(legacy=True), seeds 100000+i, max_steps 300, score = max
clipped coverage-reward per episode, mean over episodes. Registered deltas from
the runner: n_test 20, state env (no pixels — oracle bypasses perception).

  PYTHONPATH=~/brain/ishneet/diffusion_policy python -m world_tokenizer.pusht_cem_oracle
"""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np

GOAL_XY = np.array([256.0, 256.0])
GOAL_TH = np.pi / 4
POP, ELITE, ITERS = 256, 32, 4
H = 8                                                # overridden by --horizon (pt. 24)
AGENT_W = 0.0                                        # overridden by --agent-w (pt. 24)
SIGMA0, SIGMA_FLOOR = 100.0, 10.0                    # raw px (latent planner: /256)


def snap(env):
    return (tuple(env.agent.position), tuple(env.agent.velocity),
            tuple(env.block.position), env.block.angle,
            tuple(env.block.velocity), env.block.angular_velocity)


def restore(env, s):
    env.agent.position, env.agent.velocity = s[0], s[1]
    env.block.position, env.block.angle = s[2], s[3]
    env.block.velocity, env.block.angular_velocity = s[4], s[5]


def pd_env_step(env, action):
    """One env step (10 substeps of PD + physics), pusht_env.step verbatim minus reward."""
    from pymunk import Vec2d
    dt = 1.0 / env.sim_hz
    for _ in range(env.sim_hz // env.control_hz):
        acceleration = env.k_p * (action - env.agent.position) + \
            env.k_v * (Vec2d(0, 0) - env.agent.velocity)
        env.agent.velocity += acceleration * dt
        env.space.step(dt)


def pose_cost(env):
    exy = np.linalg.norm(np.array(tuple(env.block.position)) - GOAL_XY)
    eth = (env.block.angle - GOAL_TH + np.pi) % (2 * np.pi) - np.pi
    c = exy + 60.0 * abs(eth)
    if AGENT_W > 0:                                  # pt. 24 agent-shaping variant
        c += AGENT_W * np.linalg.norm(
            np.array(tuple(env.agent.position)) - np.array(tuple(env.block.position)))
    return c


def plan(env, scratch, prev, rng):
    s = snap(env)
    if prev is None:
        mean = np.full((H, 4), 256.0)
    else:
        mean = np.concatenate([prev[1:], prev[-1:]], 0)
    sigma = np.full((H, 4), SIGMA0)
    for _ in range(ITERS):
        acts = np.clip(mean + sigma * rng.standard_normal((POP, H, 4)), 0, 512)
        cost = np.empty(POP)
        for p in range(POP):
            restore(scratch, s)
            c = 0.0
            for h in range(H):
                pd_env_step(scratch, acts[p, h, :2])
                pd_env_step(scratch, acts[p, h, 2:])
                if h >= H - 2:
                    c += pose_cost(scratch) / 2
            cost[p] = c
        idx = np.argsort(cost)[:ELITE]
        el = acts[idx]
        mean = el.mean(0)
        sigma = np.maximum(el.std(0), SIGMA_FLOOR)
    return mean


def episode(seed_maxsteps):
    global H, AGENT_W
    seed, max_steps, H, AGENT_W = seed_maxsteps
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    env = PushTEnv(legacy=True)
    env.seed(seed)
    env.reset()
    scratch = PushTEnv(legacy=True)
    scratch.reset()
    rng = np.random.default_rng(seed * 7 + 1)
    prev, max_r, steps, t0, calls = None, 0.0, 0, time.perf_counter(), 0
    while steps < max_steps:
        prev = plan(env, scratch, prev, rng)
        calls += 1
        done = False
        for a in (prev[0, :2], prev[0, 2:]):
            _, r, done, _ = env.step(a)
            max_r = max(max_r, float(r))
            steps += 1
            if done or steps >= max_steps:
                break
        if done:
            break
    wall = time.perf_counter() - t0
    print(f"seed {seed}: max_reward {max_r:.3f} steps {steps} "
          f"calls {calls} wall {wall:.0f}s", flush=True)
    return {"seed": seed, "max_reward": max_r, "steps": steps,
            "plan_calls": calls, "wall_sec": wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--start-seed", type=int, default=100000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--agent-w", type=float, default=0.0)
    ap.add_argument("--out", default="/home/menlo/brain/ishneet/world-encoder/"
                                     "results/pusht/rung0_oracle.json")
    args = ap.parse_args()
    jobs = [(args.start_seed + i, args.max_steps, args.horizon, args.agent_w)
            for i in range(args.n_test)]
    with mp.Pool(args.procs) as pool:
        rows = pool.map(episode, jobs)
    score = float(np.mean([r["max_reward"] for r in rows]))
    res = {"arm": "oracle_sim", "n_test": args.n_test,
           "cem": {"pop": POP, "elite": ELITE, "iters": ITERS,
                   "horizon": args.horizon, "agent_w": args.agent_w,
                   "sigma0": SIGMA0, "sigma_floor": SIGMA_FLOOR},
           "test_mean_score": score, "episodes": rows}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"ORACLE mean max-reward over {args.n_test} eps: {score:.4f}", flush=True)
    print(f"SAVED {args.out}", flush=True)


if __name__ == "__main__":
    main()
