import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float
import catppuccin as cat

from ..constants import EMPTY_CELL, WALL_CELL, BORDER_CELL
from ..environment import (
    Environment,
    EnvParams,
    State,
    EnvCarry,
    Timestep,
)


class SimpleEnvCarry(EnvCarry): ...


CELL_A = 0.6
CELL_B = 0.7
CELL_C = 0.8
CELL_D = 0.9


class RandColors(Environment):
    def default_params(self, **kwargs: Any) -> EnvParams:
        params = EnvParams(
            height=15,
            width=11,
            view_size=5,
            num_agents=1,
            num_actions=4,
        )
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        # build the grid
        grid = jnp.full((params.height, params.width), EMPTY_CELL)
        grid = grid.at[4:15, 0:4].set(WALL_CELL)
        grid = grid.at[4:15, 7:11].set(WALL_CELL)

        grid = grid.at[0:4, 0:4].set(CELL_A)  # left room
        grid = grid.at[0:4, 7:11].set(CELL_B)  # right room

        # agent attributes
        initial_positions = jnp.asarray(
            [[params.height - 1, params.width // 2]] * params.num_agents
        )
        agent_values = jnp.linspace(0.1, 0.5, num=params.num_agents)

        return State(
            grid=grid,
            step=0,
            agents_pos=initial_positions,
            agent_values=agent_values,
            carry=SimpleEnvCarry(),
        )

    def _cell_types_and_colors(
        self, state: State
    ) -> tuple[Float[Array, " cell_types"], Integer[Array, " num_colors 3"]]:
        from ..utils import rgb

        palette = cat.PALETTE.latte.colors

        colors = jnp.asarray(
            [
                # colors for simple objects:
                rgb(palette.overlay1),  # border
                rgb(palette.subtext0),  # wall
                rgb(palette.base),  # empty
                # colors to randomly chose
                rgb(palette.peach),
                rgb(palette.red),
                rgb(palette.teal),
                rgb(palette.green),
                # colors for agents:
                rgb(palette.lavender),
                rgb(palette.maroon),
                rgb(palette.teal),
                rgb(palette.yellow),
                rgb(palette.sky),
                rgb(palette.green),
                rgb(palette.blue),
            ]
        )

        not_agents = jnp.asarray(
            (BORDER_CELL, WALL_CELL, EMPTY_CELL, CELL_A, CELL_B, CELL_C, CELL_D)
        )
        cell_types = jnp.hstack((not_agents, state.agent_values))
        return cell_types, colors

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
        new_positions: Integer[Array, "{params.num_agents} 2"],
    ) -> State:
        # simple update: update with new positions and advance step by one, leave the grid unchanged
        prev_state = timestep.state

        grid = prev_state.grid

        key_l, key_r = jax.random.split(key)
        color_l = jax.random.choice(key_l, jnp.asarray([CELL_A, CELL_B]))
        color_r = jax.random.choice(
            key_r, jnp.asarray([CELL_A, CELL_B, CELL_C, CELL_D])
        )

        grid = grid.at[0:4, 0:4].set(color_l)  # left room
        grid = grid.at[0:4, 7:11].set(color_r)  # right room

        new_state = prev_state.replace(
            grid=grid,
            step=timestep.state.step + 1,
            agents_pos=new_positions,
        )

        return new_state
