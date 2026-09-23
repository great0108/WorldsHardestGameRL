from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from train import make_eval_vec_env, prepare_runtime_caches


def _run_root(model_path: Path) -> Path:
    return model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a WHG pixel PPO model with the training environment.")
    ap.add_argument("model", help="path to PPO .zip model")
    ap.add_argument("--level", type=int, default=3, choices=range(1, 31))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--height", type=int, default=80)
    ap.add_argument("--resize-method", choices=["nearest", "bilinear", "box"], default="box")
    ap.add_argument("--delay-min", type=int, default=0)
    ap.add_argument("--delay-max", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reset-cache", default=None)
    ap.add_argument("--spatial-cache", default=None)
    ap.add_argument("--rebuild-reset-cache", action="store_true")
    ap.add_argument("--rebuild-spatial-cache", action="store_true")
    args = ap.parse_args()

    if args.episodes < 1:
        ap.error("--episodes must be >= 1")
    if args.frame_stack < 1:
        ap.error("--frame-stack must be >= 1")
    if args.frame_skip < 1:
        ap.error("--frame-skip must be >= 1")
    if args.max_steps < 1:
        ap.error("--max-steps must be >= 1")
    if args.delay_min < 0:
        ap.error("--delay-min must be >= 0")
    if args.delay_max < args.delay_min:
        ap.error("--delay-max must be >= --delay-min")

    # Match train.py's spatial reward defaults. These affect reported returns,
    # while observations/physics are shared exactly through make_eval_vec_env().
    args.step_penalty = -0.003
    args.progress_scale = 0.05
    args.death_penalty = -1.0
    args.checkpoint_bonus = 1.0
    args.coin_bonus = 1.0
    args.clear_bonus = 1.0
    args.eval_delay_min = args.delay_min
    args.eval_delay_max = args.delay_max

    model_path = Path(args.model).resolve()
    prepare_runtime_caches(args, _run_root(model_path), max_delay=args.delay_max)

    env = make_eval_vec_env(args)
    model = PPO.load(str(model_path), env=env, device=args.device)

    returns: list[float] = []
    decisions: list[int] = []
    flash_lengths: list[int] = []
    successes: list[float] = []
    delays: list[int] = []

    obs = env.reset()
    ep_return = 0.0
    ep_decisions = 0
    try:
        while len(returns) < args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            ep_decisions += 1

            if bool(dones[0]):
                info = infos[0]
                returns.append(ep_return)
                decisions.append(ep_decisions)
                flash_lengths.append(int(info.get("agent_flash_frames", 0)))
                successes.append(float(info.get("is_success", False)))
                delays.append(int(info.get("start_delay_frames", 0)))
                ep_return = 0.0
                ep_decisions = 0
    finally:
        env.close()

    print(f"Model:        {model_path}")
    print(f"Level:        {args.level}")
    print(f"Episodes:     {args.episodes}")
    print(f"Success rate: {100 * np.mean(successes):.1f}%")
    print(f"Mean reward:  {np.mean(returns):.3f}")
    print(f"Mean decisions:    {np.mean(decisions):.1f}")
    print(f"Mean Flash frames: {np.mean(flash_lengths):.1f}")
    print(f"Mean start delay:  {np.mean(delays):.1f} frames")


if __name__ == "__main__":
    main()
