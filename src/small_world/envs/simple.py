import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float

from ..environment import (
    Environment,
    EnvParams,
    State,
    EnvCarry,
    Timestep,
)
from ..constants import EMPTY_CELL, WALL_CELL
from ..utils import sample_empty_coordinates


class SimpleEnvCarry(EnvCarry): ...


class Simple(Environment):
    def default_params(self, **kwargs: Any) -> EnvParams:
        params = EnvParams(
            height=10,
            width=10,
            view_size=5,
            num_agents=1,
            num_actions=4,
        )
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        grid = jnp.full((params.height, params.width), EMPTY_CELL)
        grid = grid.at[0:2, 4].set(WALL_CELL)
        grid = grid.at[4, 0:4].set(WALL_CELL)

        initial_positions = sample_empty_coordinates(key, grid, params.num_agents)

        agent_values = jnp.linspace(0.1, 1, num=params.num_agents)

        return State(
            grid=grid,
            step=0,
            agents_pos=initial_positions,
            agent_values=agent_values,
            carry=SimpleEnvCarry(),
        )

    def _compute_rewards(
        self, params: EnvParams, state: State, key: PRNGKeyArray
    ) -> Float[Array, "{params.num_agents}"]:
        return jnp.zeros((params.num_agents))

    def _update_state(
        self,
        params: EnvParams,
        timestep: Timestep,
        actions: Integer[Scalar, "{params.num_agents}"],
        new_positions: Integer[Array, "{params.num_agents} 2"],
    ) -> State:
        # simple update: update with new positions and advance step by one, leave the grid unchanged
        prev_state = timestep.state
        new_state = prev_state.replace(
            grid=prev_state.grid,
            step=timestep.state.step + 1,
            agents_pos=new_positions,
        )

        return new_state
