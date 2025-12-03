import abc
from typing import TypeAlias, Any
from flax.struct import PyTreeNode
from flax import struct

import jax
import jax.numpy as jnp
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray


EMPTY_CELL = 0
AGENT_CELL = 1


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
    step: int = struct.field(pytree_node=False)
    agents_pos: Integer[Array, "num_agents 2"]
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
        self, params: EnvParams, grid: Grid, position: Integer[Array, "2"]
    ) -> Float[Array, "{params.view_size} {params.view_size}"]:
        vs = params.view_size

        # TODO we should fill the pad with EMPTY_CELL instead
        grid = jnp.pad(grid, pad_width=vs, mode="constant")

        # account for padding
        x, y = position[1] + vs, position[0] + vs

        observation = jax.lax.dynamic_slice(grid, (x - vs // 2, y - vs // 2), (vs, vs))

        print("***TODO!*** Include other agents in the observation")

        return observation

    def reset(self, params: EnvParams, key: PRNGKeyArray) -> Timestep:
        key_gen, key_rwd = jax.random.split(key)
        state = self._generate_problem(params, key_gen)

        positions = state.agents_pos
        observations = jax.vmap(self._get_observation, in_axes=(None, None, 0))(
            params, state.grid, positions
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

        # check collisions
        new_x, new_y = jax.lax.cond(
            grid[y, x] < 0,  # NOTE cell values < 0 are "solid material"
            lambda: (x, y),
            lambda: (new_x, new_y),
        )

        return jnp.asarray((new_x, new_y))

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
        grid = timestep.state.grid
        positions = timestep.state.agents_pos
        new_positions = jax.vmap(self._move_agent, in_axes=(None, 0, 0))(
            grid, positions, actions
        )

        new_state = self._update_state(params, timestep, actions, new_positions)

        rewards = self._compute_rewards(params, new_state, key)

        observations = jax.vmap(self._get_observation, in_axes=(None, None, 0))(
            params, new_state.grid, new_state.agents_pos
        )

        return Timestep(observations, rewards, new_state)

    def render(
        self, params: EnvParams, timestep: Timestep
    ) -> Float[Array, "{params.height} {params.width}"]:
        positions = timestep.state.agents_pos
        grid = timestep.state.grid
        print("Positions:", positions)
        grid = grid.at[positions[:, 1], positions[:, 0]].set(AGENT_CELL)

        # normalize grid to get the final image
        img = grid - grid.min()
        img /= img.max()

        return img
