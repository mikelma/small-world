# /// script
# dependencies = [
#   "pygame",
#   "tyro",
#   "jax",
#   "jaxlib",
#   "flax",
#   "catppuccin",
#   "jaxtyping",
#   "beartype",
# ]
# ///
import jax
import jax.numpy as jnp
import pygame  # type: ignore[unresolved-import]
import tyro  #  type: ignore[unresolved-import]
from dataclasses import dataclass
from typing import Tuple
from jaxtyping import install_import_hook

import sys
from pathlib import Path

# Ensure the root directory is in the path so 'src' can be found
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


with install_import_hook("src", "beartype.beartype"):
    from src.small_world.envs.adventure import Adventure


@dataclass
class Args:
    """Interactive player for the Small World Adventure environment."""

    file_name: str = "./envs-txt/adventure_M.txt"
    agent_pos: Tuple[int, int] = (17, 22)
    window_size: int = 800  # Size of the window in pixels
    fps: int = 30


def main(args: Args):
    env = Adventure()
    env_params = env.default_params(
        file_name=args.file_name, agent_init_pos=jnp.array(args.agent_pos)
    )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    render_fn = jax.jit(env.render)

    key = jax.random.key(42)
    key, reset_key = jax.random.split(key)
    timestep = reset_fn(env_params, reset_key)

    pygame.init()
    screen = pygame.display.set_mode((args.window_size, args.window_size))
    pygame.display.set_caption("Small World: Adventure Mode")
    clock = pygame.time.Clock()

    ACTION_MAP = {
        pygame.K_UP: 0,
        pygame.K_w: 0,
        pygame.K_DOWN: 1,
        pygame.K_s: 1,
        pygame.K_LEFT: 2,
        pygame.K_a: 2,
        pygame.K_RIGHT: 3,
        pygame.K_d: 3,
    }

    print("Controls: Arrow Keys or WASD to move. ESC to quit.")

    running = True
    while running:
        action = None

        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in ACTION_MAP:
                    action = ACTION_MAP[event.key]

        # Environment Step
        # Only step if a key was pressed
        if action is not None:
            key, step_key = jax.random.split(key)
            # actions array expects shape (num_agents,)
            actions = jnp.array([action])
            timestep = step_fn(step_key, env_params, timestep, actions)

        # Rendering
        img = render_fn(env_params, timestep)
        img_array = jax.device_get(img)  # Pull to CPU for pygame

        # Create surface and scale to window size
        surface = pygame.surfarray.make_surface(img_array.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (args.window_size, args.window_size))

        screen.blit(surface, (0, 0))
        pygame.display.flip()

        clock.tick(args.fps)

    pygame.quit()


if __name__ == "__main__":
    main(tyro.cli(Args))
