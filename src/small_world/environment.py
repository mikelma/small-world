import abc
from typing import TypeAlias, Any
from flax.struct import PyTreeNode

import jax 
from jaxtyping import Scalar, ScalarLike, Array, Integer, Float, PRNGKeyArray


EMPTY_CELL = 0
AGENT_CELL = 1


Grid: TypeAlias = Float[Array, "height width"]


class EnvCarry(PyTreeNode): ...


class EnvParams(PyTreeNode):
    height: int
    width: int
    num_agents: int
    num_actions: int


class State(PyTreeNode):
    grid: Grid
    step: int
    agents_pos: Integer[Array, "num_agents 2"]
    carry: EnvCarry
    

class Timestep(PyTreeNode):
    observations: Float[Array, "num_agents height width"]
    rewards: Float[Array, "num_agents"]

    state: State


class Environment(abc.ABC):

    @abc.abstractmethod
    def default_params(self, **kwargs: dict[str, Any]) -> EnvParams:
        ...

    @abc.abstractmethod
    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State:
        ...

    def reset(self, params: EnvParams, key: PRNGKeyArray) -> Timestep:
        state = self._generate_problem()

        

        
