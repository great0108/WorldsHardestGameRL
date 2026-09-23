from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pygame
from stable_baselines3 import PPO

from train import make_worker, prepare_runtime_caches


def _run_root(model_path: Path) -> Path:
    return model_path.parent.parent if model_path.parent.name == "checkpoints" else model_path.parent


def _policy_obs(stacked_hwc: np.ndarray) -> np.ndarray:
    return np.transpose(stacked_hwc, (2, 0, 1))


def _pump_quit() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _blit_frame(screen: pygame.Surface, frame: np.ndarray) -> None:
    screen.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))


def _hold_terminal(
    screen: pygame.Surface,
    frame: np.ndarray,
    *,
    status: str,
    episode: int,
    decisions: int,
    flash_frames: int,
    seconds: float,
) -> bool:
    _blit_frame(screen, frame)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 75))
    screen.blit(overlay, (0, 0))

    big = pygame.font.Font(None, 58)
    small = pygame.font.Font(None, 25)
    label = big.render(status, True, (255, 255, 255))
    screen.blit(label, label.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 12)))
    details = small.render(
        f"episode {episode}   decisions={decisions}   flash_frames={flash_frames}",
        True,
        (255, 255, 255),
    )
    screen.blit(details, details.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 32)))
    pygame.display.flip()

    end = time.perf_counter() + max(0.0, seconds)
    clock = pygame.time.Clock()
    while time.perf_counter() < end:
        if _pump_quit():
            return True
        clock.tick(60)
    return _pump_quit()


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch a PPO agent in the exact training environment.")
    ap.add_argument("model", help="path to PPO .zip model")
    ap.add_argument("--level", type=int, default=3, choices=range(1, 31))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--frame-stack", type=int, default=3)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--height", type=int, default=80)
    ap.add_argument("--resize-method", choices=["nearest", "bilinear", "box"], default="box")
    ap.add_argument("--delay-min", type=int, default=0)
    ap.add_argument("--delay-max", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--fps", type=float, default=30.0, help="real Flash-frame playback speed")
    ap.add_argument("--terminal-pause", type=float, default=1.5)
    ap.add_argument("--reset-cache", default=None)
    ap.add_argument("--spatial-cache", default=None)
    ap.add_argument("--rebuild-reset-cache", action="store_true")
    ap.add_argument("--rebuild-spatial-cache", action="store_true")
    args = ap.parse_args()

    if args.episodes < 1:
        ap.error("--episodes must be >= 1")
    if args.max_steps < 1:
        ap.error("--max-steps must be >= 1")
    if args.frame_stack < 1:
        ap.error("--frame-stack must be >= 1")
    if args.frame_skip < 1:
        ap.error("--frame-skip must be >= 1")
    if args.fps <= 0:
        ap.error("--fps must be > 0")
    if args.delay_min < 0:
        ap.error("--delay-min must be >= 0")
    if args.delay_max < args.delay_min:
        ap.error("--delay-max must be >= --delay-min")

    # make_worker() needs the same shaping fields as train.py. Reward values do
    # not affect the watched policy observation or physics.
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

    model = PPO.load(str(model_path), device=args.device)
    env = make_worker(args, 0, eval_mode=True)()
    base_env = env.unwrapped

    expected_shape = (3 * args.frame_stack, args.height, args.width)
    model_shape = tuple(model.observation_space.shape)
    if model_shape != expected_shape:
        env.close()
        raise ValueError(
            "model observation shape does not match watch settings: "
            f"model={model_shape}, requested={expected_shape}"
        )

    pygame.init()
    screen = pygame.display.set_mode((550, 400))
    pygame.display.set_caption("WHG PPO Watch — ESC to quit")
    clock = pygame.time.Clock()
    display_hz = max(1, int(round(args.fps / args.frame_skip)))

    print(f"model: {model_path}")
    print(f"observation: legacy_exact {args.width}x{args.height}, stack={args.frame_stack}")
    print(f"frame skip: {args.frame_skip}")
    print(f"start delay: {args.delay_min}..{args.delay_max} frames")
    print(f"policy: {'stochastic' if args.stochastic else 'deterministic'}")

    completed = 0
    quit_requested = False
    try:
        while completed < args.episodes and not quit_requested:
            obs, reset_info = env.reset()
            decisions = 0
            ep_return = 0.0
            delay = int(reset_info.get("start_delay_frames", 0))
            print(f"episode {completed + 1}: start delay={delay} frames ({delay / 30:.3f}s)")

            _blit_frame(screen, base_env.render())
            pygame.display.flip()

            while True:
                if _pump_quit():
                    quit_requested = True
                    break

                action, _ = model.predict(
                    _policy_obs(obs),
                    deterministic=not args.stochastic,
                )
                obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                ep_return += float(reward)
                decisions += 1

                frame = base_env.render()
                _blit_frame(screen, frame)
                pygame.display.flip()

                if terminated or truncated:
                    completed += 1
                    if info.get("is_success", False):
                        status = "SUCCESS"
                    elif info.get("death_triggered", False):
                        status = "DEAD"
                    else:
                        status = "TIMEOUT"
                    flash_frames = int(info.get("agent_flash_frames", 0))
                    print(
                        f"episode {completed}: {status}, reward={ep_return:.3f}, "
                        f"decisions={decisions}, flash_frames={flash_frames}"
                    )
                    quit_requested = _hold_terminal(
                        screen,
                        frame,
                        status=status,
                        episode=completed,
                        decisions=decisions,
                        flash_frames=flash_frames,
                        seconds=args.terminal_pause,
                    )
                    break

                clock.tick(display_hz)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
