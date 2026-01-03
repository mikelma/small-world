from .environment import Grid
from .constants import EMPTY_CELL
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool, Integer, PRNGKeyArray
from catppuccin.models import Color


def rgb(color: Color) -> Integer[Array, "3"]:
    """Converts a catppuccin `Color` object into a jax array of it's RGB values."""
    c = color.rgb
    return jnp.asarray((c.r, c.g, c.b))


def empty_cells_mask(grid: Grid) -> Bool[Array, "height width"]:
    return grid == EMPTY_CELL


def traversable_cells_mask(grid: Grid) -> Bool[Array, "height width"]:
    return grid >= EMPTY_CELL


def grid_coords(grid: Grid) -> Integer[Array, "height*width 2"]:
    coords = jnp.mgrid[: grid.shape[0], : grid.shape[1]]
    coords = coords.transpose(1, 2, 0).reshape(-1, 2)
    return coords


def sample_empty_coordinates(
    key: PRNGKeyArray, grid: Grid, num: int
) -> Integer[Array, "{num} 2"]:
    coords = grid_coords(grid)
    mask = empty_cells_mask(grid).flatten()

    # set to zero the prob of sampling a coordinate index that contains a non-empty cell
    probs = (jnp.ones_like(mask) * mask) / mask.sum()
    sampled = jax.random.choice(
        key=key,
        shape=(num,),
        a=jnp.arange(grid.shape[0] * grid.shape[1]),
        replace=False,
        p=probs,
    )
    return coords[sampled]
