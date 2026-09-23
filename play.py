from __future__ import annotations

import argparse

import numpy as np
import pygame

from whg_swf_gym.core import ACTION_TO_VECTOR, WorldsHardestGameCore
from whg_swf_gym.game_data import GameData
from whg_swf_gym.render import Renderer


VEC_TO_ACTION = {vector: action for action, vector in ACTION_TO_VECTOR.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Play the SWF-reconstructed 30-level WHG")
    ap.add_argument("--level", type=int, default=None, help="fixed level 1..30; omit for full campaign")
    args = ap.parse_args()
    if args.level is not None and not 1 <= args.level <= 30:
        ap.error("--level must be 1..30")

    data = GameData()
    core = WorldsHardestGameCore(data, level=args.level or 1, campaign=args.level is None)
    renderer = Renderer(data)

    pygame.init()
    screen = pygame.display.set_mode((550, 400))
    pygame.display.set_caption("The World's Hardest Game — SWF reconstruction")
    clock = pygame.time.Clock()

    running = True
    cleared_banner = 0
    fixed_complete = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    core.reset(level=args.level or core.level, deaths=core.deaths)
                    fixed_complete = False
                elif event.key == pygame.K_LEFTBRACKET:
                    core.set_level(max(1, core.level - 1), preserve_deaths=True)
                elif event.key == pygame.K_RIGHTBRACKET:
                    core.set_level(min(30, core.level + 1), preserve_deaths=True)

        keys = pygame.key.get_pressed()
        dx = int(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - int(keys[pygame.K_LEFT] or keys[pygame.K_a])
        dy = int(keys[pygame.K_DOWN] or keys[pygame.K_s]) - int(keys[pygame.K_UP] or keys[pygame.K_w])

        if not fixed_complete:
            info = core.step_frame(VEC_TO_ACTION[(dx, dy)])
            if info.game_cleared:
                cleared_banner = 180
                fixed_complete = args.level is not None
            elif info.level_cleared and args.level is not None:
                cleared_banner = 90
                fixed_complete = True

        frame = renderer.rgb(core)
        screen.blit(pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2))), (0, 0))

        if cleared_banner:
            font = pygame.font.Font(None, 42)
            message = "YOU WIN!  R TO RESTART" if core.level == 30 else "LEVEL COMPLETE  -  R TO RESTART"
            text = font.render(message, True, (255, 255, 255))
            rect = text.get_rect(center=(275, 200))
            pygame.draw.rect(screen, (0, 0, 0), rect.inflate(30, 20))
            screen.blit(text, rect)
            cleared_banner -= 1

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
