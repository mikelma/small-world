import jax
import jax.numpy as jnp
from typing import Any, TypeAlias
from jaxtyping import Array, Scalar, Integer, PRNGKeyArray, Float, ScalarLike, Bool
from flax import struct
import catppuccin as cat

from ..environment import (
    Environment,
    EnvParams,
    State,
    EnvCarry,
    Timestep,
    Grid,
)
from ..constants import EMPTY_CELL, BORDER_CELL
from ..utils import sample_empty_coordinates, rgb

SEA_CELL = EMPTY_CELL
LAND_CELL = 0.1
ITEM_CELL = 0.2

IntLike: TypeAlias = Integer[ScalarLike, ""]


class AdventureEnvCarry(EnvCarry):
    avail_items: Bool[Array, " num_items"]
    immunity: IntLike


class AdventureParams(EnvParams):
    items_pos: Integer[Array, "num_items 2"] = struct.field(pytree_node=True)
    map: Grid = struct.field(pytree_node=True)
    agent_init_pos: Integer[Array, "2"] = struct.field(pytree_node=True)


class Adventure(Environment):
    def default_params(self, **kwargs: Any) -> EnvParams:
        assert "num_agents" not in kwargs, (
            "changing the number of agents is not allowed in this environment"
        )

        with open(kwargs["file_name"], "r") as f:
            str_lst = [line.strip() for line in f.readlines()]

        dtype = kwargs["dtype"] if "dtype" in kwargs else jnp.float32
        n_rows, n_cols = len(str_lst), len(str_lst[0])
        grid = jnp.full((n_rows, n_cols), EMPTY_CELL, dtype=dtype)

        ascii_cells = {".": SEA_CELL, "+": LAND_CELL, "*": ITEM_CELL}

        agent_init_pos = []
        items_pos = []
        for i in range(n_rows):
            for j in range(n_cols):
                char = str_lst[i][j]
                # don't set a grid value for agents. Agent positions are handled below
                value = EMPTY_CELL if char.isalpha() else ascii_cells[char]
                grid = grid.at[i, j].set(value)
                if char.isalpha():
                    agent_init_pos.append((i, j))
                if value == ITEM_CELL:
                    items_pos.append((i, j))

        # if no agent is detect in the map file, set their initial positions
        # to (-1, -1) to later be randomly chosen in self._generate_problem
        if len(agent_init_pos) == 0:
            agent_init_pos = [-1, -1]

        params = AdventureParams(
            height=n_rows,
            width=n_cols,
            view_size=11,
            num_agents=1,
            num_actions=4,
            map=grid,
            agent_init_pos=jnp.asarray(agent_init_pos),
            items_pos=jnp.asarray(items_pos),
        )

        to_delete = ["file_name", "num_agents", "dtype"]
        for key in to_delete:
            if key in kwargs:
                del kwargs[key]
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        grid = params.map  # type: ignore (unresolved-attribute)

        agent_values = jnp.asarray([1.0])

        # use random initial positions if needed (see `self.default_params`)
        init_pos = jax.lax.cond(
            (params.agent_init_pos == -1).any(),  # type: ignore (unresolved-attribute)
            lambda: sample_empty_coordinates(key, grid, 1)[0],
            lambda: params.agent_init_pos,  # type: ignore (unresolved-attribute)
        )

        num_items = params.items_pos.shape[0]  # type: ignore (unresolved-attribute)
        return State(
            grid=grid,
            step=0,
            agents_pos=init_pos[None, :],
            agent_values=agent_values,
            carry=AdventureEnvCarry(
                avail_items=jnp.full((num_items,), True), immunity=0
            ),
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
        new_positions: Integer[Array, "{params.num_agents} 2"],
    ) -> State:
        carry = timestep.state.carry
        items_pos = params.items_pos  # type: ignore (unresolved-attribute)
        avail_items = carry.avail_items  # type: ignore (unresolved-attribute)
        grid = timestep.state.grid
        immunity = carry.immunity  # type: ignore (unresolved-attribute)

        # check if the agent is over an item
        item_mask = (
            (new_positions[0][0] == items_pos[:, 0])
            & (new_positions[0][1] == items_pos[:, 1])
            & avail_items
        )

        def _on_true():
            item_id = jnp.argmax(item_mask)
            new_avail = avail_items.at[item_id].set(False)

            item_pos = items_pos[item_id]
            new_grid = grid.at[item_pos[0], item_pos[1]].set(LAND_CELL)
            return new_avail, new_grid, immunity + 10

        def _on_false():
            return avail_items, grid, immunity  # old values from prev step

        over_item = item_mask.sum() > 0
        new_avail, new_grid, part_immunity = jax.lax.cond(
            over_item, _on_true, _on_false
        )

        # check if the agent is over the water
        over_sea = grid[new_positions[0][0], new_positions[0][1]] == SEA_CELL
        new_immunity = jax.lax.cond(
            over_sea,
            lambda: jnp.clip(immunity - 1, min=0),
            lambda: part_immunity,
        )

        # check immunity (condition of death)
        old_agent_pos = timestep.state.agents_pos
        already_over_sea = grid[old_agent_pos[0][0], old_agent_pos[0][1]] == SEA_CELL
        new_positions = jax.lax.cond(
            already_over_sea & (immunity == 0),
            lambda: old_agent_pos,
            lambda: new_positions,
        )

        new_state = timestep.state.replace(
            grid=new_grid,
            step=timestep.state.step + 1,
            agents_pos=new_positions,
            carry=carry.replace(avail_items=new_avail, immunity=new_immunity),
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
                rgb(palette2.green),  # land
                rgb(palette2.blue),  # sea
                rgb(palette.mauve),  # item
                # colors for agents:
                # NOTE: there're colors up to 4 agents, after that,
                # the rest of agents will have the last color in this list
                rgb(palette.red),
                rgb(palette.maroon),
                rgb(palette.teal),
                rgb(palette.peach),
            ]
        )

        simple_types = jnp.asarray((BORDER_CELL, LAND_CELL, SEA_CELL, ITEM_CELL))
        cell_types = jnp.hstack((simple_types, state.agent_values))
        return cell_types, colors
