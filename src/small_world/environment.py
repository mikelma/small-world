import abc
from typing import TypeAlias, Any
from flax.struct import PyTreeNode
from flax import struct

import jax
import jax.numpy as jnp
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray

import catppuccin as cat

from .constants import BORDER_CELL, EMPTY_CELL, WALL_CELL


Grid: TypeAlias = Float[Array, "height width"]
IntLike: TypeAlias = Integer[ScalarLike, ""]


class EnvCarry(PyTreeNode): ...


class EnvParams(PyTreeNode):
    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)
    num_agents: int = struct.field(pytree_node=False)
    num_actions: int = struct.field(pytree_node=False)
    view_size: int = struct.field(pytree_node=False)


class State(PyTreeNode):
    grid: Grid
    step: IntLike
    agents_pos: Integer[Array, "num_agents 2"]
    agent_values: Float[Array, "num_agents"]
    carry: EnvCarry


class Timestep(PyTreeNode):
    observations: Float[Array, "num_agents height width"]
    rewards: Float[Array, "num_agents"]

    state: State


class Environment(abc.ABC):
    @abc.abstractmethod
    def default_params(self, **kwargs: dict[str, Any]) -> EnvParams: ...

    @abc.abstractmethod
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State: ...

    @abc.abstractmethod
    def _compute_rewards(
        self, params: EnvParams, state: State, key: PRNGKeyArray
    ) -> Float[Array, "{params.num_agents}"]: ...

    def _get_observation(
        self, params: EnvParams, state: State, position: Integer[Array, "2"]
    ) -> Float[Array, "{params.view_size} {params.view_size}"]:
        vs = params.view_size

        # add all agent values to the grid
        positions = state.agents_pos
        grid = state.grid.at[positions[:, 0], positions[:, 1]].set(state.agent_values)

        # ensure that the value of the cell in the position of this agent is
        # the cell value that corresponds to this agent (relevant when more than
        # one agent are in the same cell).
        mask = (positions == position).all(axis=1).astype(int)
        agent_value = (state.agent_values * mask).sum()
        grid = grid.at[position[0], position[1]].set(agent_value)

        # pad the grid with border cells
        grid = jnp.pad(grid, pad_width=vs, mode="constant", constant_values=BORDER_CELL)

        # account for padding
        x, y = position[1] + vs, position[0] + vs

        observation = jax.lax.dynamic_slice(grid, (y - vs // 2, x - vs // 2), (vs, vs))

        return observation

    def reset(self, params: EnvParams, key: PRNGKeyArray) -> Timestep:
        key_gen, key_rwd = jax.random.split(key)
        state = self._generate_problem(params, key_gen)

        positions = state.agents_pos
        observations = jax.vmap(self._get_observation, in_axes=(None, None, 0))(
            params, state, positions
        )

        rewards = self._compute_rewards(params, state, key_rwd)

        return Timestep(
            observations,
            rewards,
            state,
        )

    def _move_agent(self, grid: Grid, position: Integer[Array, "2"], action: IntLike):
        def _make_move(x, y):
            return jax.lax.switch(
                action,
                (
                    lambda: (x, y - 1),  # up
                    lambda: (x, y + 1),  # down
                    lambda: (x - 1, y),  # left
                    lambda: (x + 1, y),  # right
                ),
            )

        x, y = position[1], position[0]

        # execute a movement action if the action is an integer between 0 and 3
        new_x, new_y = jax.lax.cond(
            action < 4, lambda: _make_move(x, y), lambda: (x, y)
        )

        # check boundaries
        new_x, new_y = (
            jnp.clip(new_x, 0, grid.shape[1] - 1),
            jnp.clip(new_y, 0, grid.shape[0] - 1),
        )

        # check collisions
        new_x, new_y = jax.lax.cond(
            grid[new_y, new_x] < 0,  # NOTE cell values < 0 are "solid material"
            lambda: (x, y),
            lambda: (new_x, new_y),
        )

        return jnp.asarray((new_y, new_x))

    @abc.abstractmethod
    def _update_state(
        self,
        params: EnvParams,
        timestep: Timestep,
        actions: Integer[Scalar, "{params.num_agents}"],
        new_positions: Integer[Array, "{params.num_agents} 2"],
    ) -> State: ...

    def step(
        self,
        key: PRNGKeyArray,
        params: EnvParams,
        timestep: Timestep,
        actions: Integer[Array, "{params.num_agents}"],
    ) -> Timestep:
        # move all agents in parallel
        grid = timestep.state.grid
        positions = timestep.state.agents_pos
        new_positions = jax.vmap(self._move_agent, in_axes=(None, 0, 0))(
            grid, positions, actions
        )

        new_state = self._update_state(params, timestep, actions, new_positions)

        rewards = self._compute_rewards(params, new_state, key)

        observations = jax.vmap(self._get_observation, in_axes=(None, None, 0))(
            params, new_state, new_state.agents_pos
        )

        return Timestep(observations, rewards, new_state)

    def _default_cell_color(self, cell: Scalar, state: State) -> Integer[Array, "3"]:
        from .utils import rgb

        palette = cat.PALETTE.latte.colors

        # specific colors for the agents
        agent_colors = jnp.asarray(
            [
                # colors for simple objects:
                rgb(palette.overlay1),  # border
                rgb(palette.subtext0),  # wall
                rgb(palette.base),  # empty
                # colors for agents:
                # NOTE: there're colors up to 10 agents, after that,
                # the rest of agents will have the last color in this list
                rgb(palette.lavender),
                rgb(palette.maroon),
                rgb(palette.teal),
                rgb(palette.yellow),
                rgb(palette.sky),
                rgb(palette.green),
                rgb(palette.peach),
                rgb(palette.sapphire),
                rgb(palette.red),
                rgb(palette.blue),
            ]
        )

        simple_types = jnp.asarray((BORDER_CELL, WALL_CELL, EMPTY_CELL))
        cell_types = jnp.hstack((simple_types, state.agent_values))

        # calculate distance between the current cell and all known agent values
        dists = jnp.abs(cell_types - cell)

        # find the closest match
        closest_idx = jnp.argmin(dists)
        return agent_colors[closest_idx]

    def grid_to_rgb(self, grid: Grid, state: State) -> Integer[Array, "height width 3"]:
        flat_grid = grid.ravel()
        colored = jax.vmap(self._default_cell_color, in_axes=(0, None))(
            flat_grid, state
        )
        img = colored.reshape(*grid.shape, 3)
        return img

    def render(
        self, params: EnvParams, timestep: Timestep, scale_factor: int = 1
    ) -> Integer[Array, "{params.height*scale_factor} {params.width*scale_factor} 3"]:
        positions = timestep.state.agents_pos
        grid = timestep.state.grid

        y_coords = positions[:, 0]
        x_coords = positions[:, 1]

        grid = grid.at[y_coords, x_coords].set(timestep.state.agent_values)

        img = self.grid_to_rgb(grid, timestep.state)

        if scale_factor > 1:
            img = jnp.repeat(img, scale_factor, axis=0)
            img = jnp.repeat(img, scale_factor, axis=1)

        return img
