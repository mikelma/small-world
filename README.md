# SmallWorld 🐜

SmallWorld is a simple and fast (multi-agent) Reinforcement Learning (RL) environment library in [JAX](https://docs.jax.dev/en/latest/index.html) ⚡.
It contains several ready-to-use environments and utilities for easily creating new ones 🏗️.

SmallWorld environments consist of grid worlds, where each cell in the grid has a scalar value in the [-1, 1] range.
This value determines the cell type: negative values correspond to solid (impenetrable) cells, such as walls or obstacles; zero denotes floor cells; and positive values represent other entities, such as items, collectible objects, or agents.
Different agents may be assigned different cell values (always positive), which makes them distinguishable from one another.

**Fatures ✨:**
- **Simple to use:**
  + Observations are agent-centric windows of continuous 2D matrices (in the [-1, 1] range).
  + Gymnasium-like environment API.
  + Start using SmallWorld in minutes thanks to its many ready-to-use environments.
- **Simple to build.** SmallWorld has been thought to be extended with new environments from the beginning.
- **Multi-agent:** Code once, and use your environment for single or multi-agent RL research.
- **It's JAX!** Use your favorite super-fast RL implementation and get impressive performance numbers.

**Few important notes ⚠️:** SmallWorld has some important distinctions compared to other grid worlds:
- The environment is rendered in RGB colors for humans, but cells in the environment's grid (2D matrix) are scalar values in the [-1, 1] range, not RGB colors.
- All agents act simultaneously in multi-agent environments by default. This can be changed for some multi-agent scenarios, but simultaneous actions keep environments faster and simpler.
- Two agents can occupy the same cell. In this case, each agent will observe itself as being at the center of its observation.

## Getting started 🚀

If using [`uv`](https://docs.astral.sh/uv/), run the following line to install the latest (`main` branch) SmallWorld version,
```bash
uv add git+https://github.com/mikelma/small-world.git
```

You can now use SmallWorld in your `uv` project!
SmallWorld uses a similar environment API to [xland-minigrid](https://github.com/dunnolab/xland-minigrid/).
For example,
```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from small_world.envs.simple import Simple
# from small_world.envs.scw import SimpleContinualWorld

key = jax.random.key(42)

env = Simple()
env_params = env.default_params(num_agents=4)

# env = SimpleContinualWorld()
# env_params = env.default_params(num_steps=100)

reset = jax.jit(env.reset)
step = jax.jit(env.step)
render = jax.jit(env.render)

key, key_reset = jax.random.split(key)
timestep = reset(env_params, key)

for _ in range(100):
    plt.cla()
    plt.imshow(render(env_params, timestep))
    plt.pause(0.02)

    key, key_step, key_act = jax.random.split(key, 3)

    # sample a random action with uniform probability
    actions = jax.random.randint(
        key_act, (env_params.num_agents,), 0, env_params.num_actions
    )

    timestep = jax.jit(step)(key_step, env_params, timestep, actions)
```

## Default environments 🌍

| Name                                                                                                 | Description                                                                                                                                      |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| [Simple](https://github.com/mikelma/small-world/blob/main/src/small_world/envs/simple.py)            | Mostly empty world with a small room in the top left corner.                                                                                     |
| [FromMap](https://github.com/mikelma/small-world/blob/main/src/small_world/envs/from_map.py)         | Loads an environment definition from an ASCII text file. See [envs-txt](https://github.com/mikelma/small-world/tree/main/envs-txt) for examples. |
| [Adventure](https://github.com/mikelma/small-world/blob/main/src/small_world/envs/adventure.py)      | Large environment with islands and water. The agent has a maximum survival time in water that resets every time it reaches land.                 |
| [RandColors](https://github.com/mikelma/small-world/blob/main/src/small_world/envs/randcolors.py)    | Large corridor with two rooms at the end. The cell types of the rooms (floor color) change at every timestep with uniform probability.          |
| [SimpleContinualWorld](https://github.com/mikelma/small-world/blob/main/src/small_world/envs/scw.py) | Non-stationary, reset-free, environment for continual RL.                                                                                        |

## Creating new environments 🏗️

**TODO**

## License

SmallWorld is distributed under the terms of the GLPv3 license. See [LICENSE](./LICENSE) for more details.
