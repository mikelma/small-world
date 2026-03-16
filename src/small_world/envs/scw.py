import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float
import catppuccin as cat

from ..environment import (
    Environment,
    EnvParams,
    State,
    EnvCarry,
    Timestep,
    IntLike,
    Grid,
    ScalarLike,
)
from ..constants import EMPTY_CELL, WALL_CELL, BORDER_CELL
from ..utils import sample_coordinates, rgb, empty_cells_mask


GOAL_CELL = float(-0.6)


class SCWCarry(EnvCarry):
    key_procgen: PRNGKeyArray


class SCWParams(EnvParams):
    probs_wall: Float[Array, " num_timesteps"]
    probs_floor: Float[Array, " num_timesteps"]


def place_random_walls(key: PRNGKeyArray, grid: Grid, prob: ScalarLike, agents_pos: Integer[Array, "num_agents 2"]) -> Grid:
    mask_empty = empty_cells_mask(grid)
    # check that we don't place a wall in any of the agents' positions
    mask_empty = mask_empty.at[agents_pos[:, 0], agents_pos[:, 1]].set(False)
    mask_bern = jax.random.bernoulli(
        key,
        p=prob,
        shape=grid.shape[0] * grid.shape[1]
    ).reshape(grid.shape)
    grid = jnp.where(mask_bern * mask_empty, jnp.full_like(grid, WALL_CELL), grid)
    return grid


def remove_random_walls(key: PRNGKeyArray, grid: Grid, prob: ScalarLike) -> Grid:
    mask_walls = grid == WALL_CELL
    mask_bern = jax.random.bernoulli(
        key,
        p=prob,
        shape=grid.shape[0] * grid.shape[1]
    ).reshape(grid.shape)
    grid = jnp.where(mask_bern * mask_walls, jnp.full_like(grid, EMPTY_CELL), grid)
    return grid


class SimpleContinualWorld(Environment):
    def default_params(self, **kwargs: Any) -> EnvParams:
        # probs_wall = jnp.full((10,), 0.1)
        # probs_floor = jnp.full((10,), 0.1)
        probs_wall = jnp.asarray([0.1, 0.0001])
        probs_floor = jnp.asarray([0.0005, 0.0005])

        params = SCWParams(
            height=20,
            width=20,
            view_size=7,
            num_agents=1,
            num_actions=4,
            probs_wall=probs_wall,
            probs_floor=probs_floor,
        )
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        grid = jnp.full((params.height, params.width), EMPTY_CELL)

        # generate the key for the procedural generation in later steps
        key, key_procgen = jax.random.split(key)

        # all agents start on the top left corner
        init_pos = jnp.zeros((params.num_agents, 2), dtype=int)
        # place some wall cells in agents' starting positions to make placing the
        # goal and objects easier on the grid. This is later replaced with an empty
        # cell at the end of this function.
        grid = grid.at[init_pos[:, 0], init_pos[:, 1]].set(WALL_CELL)

        # randomly place wall cells
        key, _key = jax.random.split(key)
        grid = place_random_walls(key=_key, grid=grid,
                                  prob=params.probs_wall[0], agents_pos=init_pos)

        # place the goal in an empty cell
        key, _key = jax.random.split(key)
        mask = empty_cells_mask(grid)
        mask = mask.at[init_pos[:, 0], init_pos[:, 1]].set(False)
        pos_goal = sample_coordinates(_key, grid, mask, 1)
        grid = grid.at[pos_goal[:, 0], pos_goal[:, 1]].set(GOAL_CELL)

        # make sure that the agents' starting position is clear
        grid = grid.at[init_pos[:, 0], init_pos[:, 1]].set(EMPTY_CELL)

        agent_values = jnp.linspace(0.1, 1, num=params.num_agents)

        return State(
            grid=grid,
            step=0,
            agents_pos=init_pos,
            agent_values=agent_values,
            carry=SCWCarry(key_procgen=key_procgen),
        )

    def _compute_rewards(
        self, params: EnvParams, state: State, key: PRNGKeyArray
    ) -> Float[Array, " {params.num_agents}"]:
        return jnp.zeros((params.num_agents))

    def _update_state(
        self,
        key: PRNGKeyArray,
        params: EnvParams,
        timestep: Timestep,
        actions: Integer[Scalar, " {params.num_agents}"],
        new_positions: Integer[Array, " {params.num_agents} 2"],
    ) -> State:
        prev_state = timestep.state
        prev_carry = prev_state.carry

        n_step = timestep.state.step + 1

        # NOTE all keys used for procgen in this function must use keys derived from the one in the carry
        key_procgen, key_empty, key_walls = jax.random.split(prev_carry.key_procgen, num=3)

        # add new wall cells
        grid = remove_random_walls(key=key_empty, grid=prev_state.grid, prob=params.probs_floor[n_step])
        grid = place_random_walls(key=key_walls, grid=grid,
                                  prob=params.probs_wall[n_step], agents_pos=new_positions)

        # simple update: update with new positions and advance step by one, leave the grid unchanged
        new_state = prev_state.replace(
            grid=grid,
            step=n_step,
            agents_pos=new_positions,
            carry=SCWCarry(key_procgen=key_procgen)
        )

        return new_state

    def _cell_types_and_colors(
        self, state: State
    ) -> tuple[Float[Array, " cell_types"], Integer[Array, " num_colors 3"]]:
        palette = cat.PALETTE.latte.colors
        palette2 = cat.PALETTE.frappe.colors

        colors = jnp.asarray(
            [
                # colors for simple objects:
                rgb(palette.overlay1),  # border
                rgb(palette2.green),  # goal cell
                rgb(palette.base),  # empty
                # colors for agents:
                # NOTE: there're colors up to 4 agents, after that,
                # the rest of agents will have the last color in this list
                rgb(palette.lavender),
                rgb(palette.red),
                rgb(palette.maroon),
                rgb(palette.teal),
                rgb(palette.peach),
            ]
        )

        simple_types = jnp.asarray((BORDER_CELL, GOAL_CELL, EMPTY_CELL))
        cell_types = jnp.hstack((simple_types, state.agent_values))
        return cell_types, colors
