import jax
import jax.numpy as jnp
from typing import Any
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float
from flax import struct

from ..environment import (
    Environment,
    EnvParams,
    State,
    EnvCarry,
    Timestep,
    Grid,
)
from ..constants import EMPTY_CELL, WALL_CELL
from ..utils import empty_cells_mask, sample_empty_coordinates


class SimpleEnvCarry(EnvCarry): ...


class FromMapParams(EnvParams):
    map: Grid = struct.field(pytree_node=True)
    agents_init_pos: Integer[Array, "num_agents 2"] = struct.field(pytree_node=True)


class FromMap(Environment):
    def default_params(self, **kwargs: Any) -> EnvParams:
        with open(kwargs["file_name"], "r") as f:
            str_lst = [line.strip() for line in f.readlines()]

        n_rows, n_cols = len(str_lst), len(str_lst[0])
        grid = jnp.full((n_rows, n_cols), EMPTY_CELL)

        ascii_cells = {".": EMPTY_CELL, "+": WALL_CELL}

        agents_init_pos = []
        for i in range(n_rows):
            for j in range(n_cols):
                char = str_lst[i][j]
                # don't set a grid value for agents. Agent positions are handled below
                value = EMPTY_CELL if char.isalpha() else ascii_cells[char]
                grid = grid.at[i, j].set(value)
                # log agent's initial position
                if char.isalpha():
                    agents_init_pos.append((i, j))

        # if no agent is detect in the map file, set their initial positions
        # to (-1, -1) to later be randomly chosen in self._generate_problem
        if len(agents_init_pos) == 0:
            n = kwargs["num_agents"] if "num_agents" in kwargs else 1
            agents_init_pos = [[-1, -1]] * n

        params = FromMapParams(
            height=n_rows,
            width=n_cols,
            view_size=5,
            num_agents=len(agents_init_pos),
            num_actions=4,
            map=grid,
            agents_init_pos=jnp.asarray(agents_init_pos),
        )
        del kwargs["file_name"]
        if "num_agents" in kwargs:
            del kwargs["num_agents"]
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        grid = params.map

        agent_values = jnp.linspace(0.1, 1, num=params.num_agents)

        # use random initial positions if needed (see `self.default_params`)
        init_pos = jax.lax.cond(
            (params.agents_init_pos == -1).any(),
            lambda: sample_empty_coordinates(key, grid, params.num_agents),
            lambda: params.agents_init_pos,
        )

        return State(
            grid=grid,
            step=0,
            agents_pos=init_pos,
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
