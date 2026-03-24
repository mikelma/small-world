import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float, Bool
import catppuccin as cat
from flax.struct import PyTreeNode

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


GOAL_CELL = float(1.0)


class EnvConditions(PyTreeNode):
    prob_wall: ScalarLike
    prob_floor: ScalarLike
    wind: ScalarLike
    wind_dir: IntLike


class SCWCarry(EnvCarry):
    key_procgen: PRNGKeyArray
    tgt_pos: Integer[Array, "2"]
    conditions: EnvConditions
    goal_reset: Bool[ScalarLike, ""]


class SCWParams(EnvParams):
    init_conditions: EnvConditions
    num_steps: IntLike

    wind_freq: ScalarLike
    wind_max: ScalarLike


def place_random_walls(
    key: PRNGKeyArray,
    grid: Grid,
    prob: ScalarLike,
    agents_pos: Integer[Array, "num_agents 2"],
) -> Grid:
    mask_empty = empty_cells_mask(grid)
    # check that we don't place a wall in any of the agents' positions
    mask_empty = mask_empty.at[agents_pos[:, 0], agents_pos[:, 1]].set(False)
    mask_bern = jax.random.bernoulli(
        key, p=prob, shape=grid.shape[0] * grid.shape[1]
    ).reshape(grid.shape)
    grid = jnp.where(mask_bern * mask_empty, jnp.full_like(grid, WALL_CELL), grid)
    return grid


def remove_random_walls(key: PRNGKeyArray, grid: Grid, prob: ScalarLike) -> Grid:
    mask_walls = grid == WALL_CELL
    mask_bern = jax.random.bernoulli(
        key, p=prob, shape=grid.shape[0] * grid.shape[1]
    ).reshape(grid.shape)
    grid = jnp.where(mask_bern * mask_walls, jnp.full_like(grid, EMPTY_CELL), grid)
    return grid


class SimpleContinualWorld(Environment):
    def default_params(self, **kwargs: Any) -> EnvParams:
        assert "num_steps" in kwargs, "Missing required argument `num_steps`"

        height, width = 32, 32
        params = SCWParams(
            height=height,
            width=width,
            view_size=7,
            num_agents=1,
            num_actions=4,
            init_conditions=EnvConditions(
                prob_wall=0.1,
                prob_floor=0.005,
                wind=0.5,
                wind_dir=2,  # left action (wind from right)
            ),
            wind_freq=3,
            wind_max=0.15,
            prob_wall=0.01 / (height * width),
            prob_floor=0.02 / (height * width),
            num_steps=kwargs["num_steps"],
        )
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        grid = jnp.full((params.height, params.width), EMPTY_CELL)

        # generate the key for the procedural generation in later steps
        key, key_procgen = jax.random.split(key)

        # all agents start on the top left corner
        init_pos = jnp.zeros((params.num_agents, 2), dtype=int)

        # randomly place wall cells
        key, _key = jax.random.split(key)
        grid = place_random_walls(
            key=_key,
            grid=grid,
            prob=params.init_conditions.prob_wall,
            agents_pos=init_pos,
        )

        # place the goal in an empty cell
        key, _key = jax.random.split(key)
        mask = empty_cells_mask(grid)
        mask = mask.at[init_pos[:, 0], init_pos[:, 1]].set(False)
        pos_goal = sample_coordinates(_key, grid, mask, 1)[0]
        grid = grid.at[pos_goal[0], pos_goal[1]].set(GOAL_CELL)

        agent_values = jnp.linspace(0.1, 0.5, num=params.num_agents)

        return State(
            grid=grid,
            step=0,
            agents_pos=init_pos,
            agent_values=agent_values,
            carry=SCWCarry(
                key_procgen=key_procgen,
                conditions=params.init_conditions,
                tgt_pos=pos_goal,
                goal_reset=False,
            ),
        )

    def _compute_rewards(
        self, params: EnvParams, state: State, key: PRNGKeyArray
    ) -> Float[Array, " {params.num_agents}"]:
        return (state.agents_pos == state.carry.tgt_pos).all(1).astype(float)

    def _update_conditions(
        self, conds: EnvConditions, tstep: IntLike, params: EnvParams
    ) -> EnvConditions:
        wind = params.wind_max * -jnp.cos((params.wind_freq * tstep / params.num_steps) * 2 * jnp.pi)
        wind = (wind + params.wind_max)  / 2  # scale to [0, wind_max]
        wind *= tstep / params.num_steps
        # wind = jnp.maximum(wind, jnp.asarray(0.0))
        return conds.replace(
            prob_wall=params.prob_wall,
            prob_floor=params.prob_floor,
            wind=wind,
        )

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
        prev_conds = prev_state.carry.conditions

        n_step = timestep.state.step + 1

        # NOTE all keys used for procgen in this function must use keys derived from the one in the carry
        key_procgen, key_empty, key_walls, key_wind, key_goal = jax.random.split(
            prev_carry.key_procgen, num=5
        )

        # add new wall cells and remove some
        grid = remove_random_walls(
            key=key_empty, grid=prev_state.grid, prob=prev_conds.prob_floor
        )
        grid = place_random_walls(
            key=key_walls,
            grid=grid,
            prob=prev_conds.prob_wall,
            agents_pos=new_positions,
        )

        # update agents' positions depending on the wind
        pos_moved = jax.vmap(self._move_agent, in_axes=(None, 0, None))(
            grid, new_positions, prev_conds.wind_dir
        )
        mask_moves = jax.random.bernoulli(
            key_wind, prev_conds.wind, (params.num_agents,)
        )
        new_positions = (~mask_moves) * new_positions + mask_moves * pos_moved

        # generate a candidate position for the new goal
        mask = empty_cells_mask(grid)
        mask = mask.at[new_positions[:, 0], new_positions[:, 1]].set(False)
        new_goal = sample_coordinates(key_goal, grid, mask, 1)[0]
        # update goal's position if reached in this new state
        tgt_pos = jax.lax.cond(
            prev_carry.goal_reset, lambda: new_goal, lambda: prev_carry.tgt_pos
        )
        # update the grid with the new goal
        grid = grid.at[tgt_pos[0], tgt_pos[1]].set(GOAL_CELL)
        prev_goal_cell = jax.lax.cond(
            prev_carry.goal_reset, lambda: EMPTY_CELL, lambda: GOAL_CELL
        )
        grid = grid.at[prev_carry.tgt_pos[0], prev_carry.tgt_pos[1]].set(prev_goal_cell)

        # update environment's conditions
        conditions = self._update_conditions(prev_carry.conditions, n_step, params)

        # check if any agent has reached the objective, request goal reset in the next step in that case
        goal_reset = (new_positions == prev_carry.tgt_pos).all(1).any()

        # simple update: update with new positions and advance step by one, leave the grid unchanged
        new_state = prev_state.replace(
            grid=grid,
            step=n_step,
            agents_pos=new_positions,
            carry=prev_carry.replace(
                key_procgen=key_procgen,
                conditions=conditions,
                tgt_pos=tgt_pos,
                goal_reset=goal_reset,
            ),
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
