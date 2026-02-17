from gymnasium import register
from ray.tune import register_env

from .MNISTExib import MNISTExib

# Register MNISTExib-v0 Gymnasium env
register(
    id="MNISTExib-v0",  # environment id in Gymnasium
    entry_point="envs.MNISTExib:MNISTExib",  # module_name:class_name
)
# Register a simplified version of MNISTExib-v0 Gymnasium env where all digits are distinguishable
register(
    id="simpleMNISTExib-v0",  # environment id in Gymnasium
    entry_point="envs.MNISTExib:MNISTExib",  # module_name:class_name
)


# Register MNISTExib-v0 Ray env
register_env("MNISTExib-v0", lambda env_config: MNISTExib(**env_config))

# Register simplified (i.e. no equal digits) MNISTExib-v0 Ray env
register_env("simpleMNISTExib-v0", lambda env_config: MNISTExib(**env_config))
