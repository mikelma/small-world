import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float

from ..environment import Environment, EnvParams, State, EnvCarry, Timestep


class SimpleEnvCarry(EnvCarry): ...


class Simple(Environment):
    def default_params(self, **kwargs: dict[str, Any]) -> EnvParams:
        params = EnvParams(
            height=10, width=10, view_size=5, num_agents=1, num_actions=4
        )
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        grid = jnp.zeros((params.height, params.width))
        # grid = (
        #     jnp.arange(params.height * params.width, dtype=float).reshape(
        #         params.height, params.width
        #     )
        #     / 10
        # )

        return State(
            grid=grid,
            step=0,
            agents_pos=jnp.asarray((5, 5), dtype=int).reshape(1, -1),
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
