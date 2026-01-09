# /// script
# dependencies = [
#   "tyro",
#   "noise",
# ]
# ///

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import noise
import tyro


@dataclass
class MapConfig:
    """Procedural ASCII Island Generator."""

    width: int = 80
    """Width of the map"""
    height: int = 40
    """Height of the map"""
    scale: float = 15.0
    """Scale of the noise (lower = larger features)"""
    threshold: float = 0.1
    """Landmass density (higher = less land)"""
    octaves: int = 6
    """Level of detail in the coastline"""
    seed: Optional[int] = None
    """Random seed for generation"""
    output: Optional[Path] = None
    """Optional file path to save the map"""


def generate_map(cfg: MapConfig) -> str:
    seed = cfg.seed or random.randint(0, 1000)
    lines = []

    center_x, center_y = cfg.width / 2, cfg.height / 2
    max_dist = (center_x**2 + center_y**2) ** 0.5

    for y in range(cfg.height):
        row = ""
        for x in range(cfg.width):
            # Generate Perlin noise
            val = noise.pnoise2(
                x / cfg.scale, y / cfg.scale, octaves=cfg.octaves, base=seed
            )

            # Apply radial gradient to force sea at edges (Island effect)
            dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 / max_dist
            val -= dist * 0.5  # Adjust 0.5 to make islands larger/smaller

            # Thresholding for ASCII characters
            row += "+" if val > (cfg.threshold - 0.5) else "."
        lines.append(row)

    return "\n".join(lines)


def main(cfg: MapConfig) -> None:
    ascii_map = generate_map(cfg)

    # Print to terminal
    print(ascii_map)

    # Save to file if requested
    if cfg.output:
        cfg.output.write_text(ascii_map)
        print(f"\nMap saved to {cfg.output}")


if __name__ == "__main__":
    cfg = tyro.cli(MapConfig)
    main(cfg)
