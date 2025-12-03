import abc
from typing import TypeAlias, Any
from flax.struct import PyTreeNode

import jax
import jax.numpy as jnp
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray


EMPTY_CELL = 0
AGENT_CELL = 1


Grid: TypeAlias = Float[Array, "height width"]
IntLike: TypeAlias = Integer[ScalarLike, ""]


class EnvCarry(PyTreeNode): ...


class EnvParams(PyTreeNode):
    height: int
    width: int
    num_agents: int
    num_actions: int
    view_size: int


class State(PyTreeNode):
    grid: Grid
    step: IntLike
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

    def get_observation(
        self, params: EnvParams, grid: Grid, position: Integer[Array, "2"]
    ) -> Float[Array, "{params.view_size} {params.view_size}"]:
        vs = params.view_size

        # TODO we should fill the pad with EMPTY_CELL instead
        grid = jnp.pad(grid, pad_width=vs, mode="constant")

        # account for padding
        x, y = position[1] + vs, position[0] + vs

        observation = jax.lax.dynamic_slice(grid, (x - vs // 2, y - vs // 2), (vs, vs))

        return observation

    def reset(self, params: EnvParams, key: PRNGKeyArray) -> Timestep:
        key_gen, key_rwd = jax.random.split(key)
        state = self._generate_problem(params, key_gen)

        positions = state.agents_pos
        observations = jax.vmap(self.get_observation, in_axes=(None, None, 0))(
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

        return new_x, new_y

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

        quit()
