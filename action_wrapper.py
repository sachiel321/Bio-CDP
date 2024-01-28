"""Environment wrapper that converts environments to use canonical action specs.

This only affects action specs of type `specs.BoundedArray`.

For bounded action specs, we refer to a canonical action spec as the bounding
box [-1, 1]^d where d is the dimensionality of the spec. So the shape and dtype
of the spec is unchanged, while the maximum/minimum values are set to +/- 1.
"""

import dm_env
import numpy as np
import tree
from dm_env import specs

from dm_env_wrappers._src import base


class CanonicalSpecWrapperv2(base.EnvironmentWrapper):
    """Wrapper which converts environments to use canonical action specs.

    This only affects action specs of type `specs.BoundedArray`.

    For bounded action specs, we refer to a canonical action spec as the bounding
    box [-1, 1]^d where d is the dimensionality of the spec. So the shape and
    dtype of the spec is unchanged, while the maximum/minimum values are set
    to +/- 1.
    """

    def __init__(self, environment: dm_env.Environment, clip: bool = False):
        super().__init__(environment)
        self._action_spec = environment.action_spec()
        self._clip = clip
        self._clip_action_spec = np.array([
            [-0.174533, 0.174533],
            [-0.488692, 0.488692],
            [0, 0],
            [0, 1.22173],
            [0, 0.20944],
            [0, 0.698132],
            [0, 1.5708],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [0,0],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [-0.7605, 0.4605],
            [0, 0.06],
            [-0.174533, 0.174533],
            [-0.488692, 0.488692],
            [0, 0],
            [0, 1.22173],
            [0, 0.20944],
            [0, 0.698132],
            [0, 1.5708],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [0,0],
            [0, 0.349066],
            [0, 1.5708],
            [0, 3.1415],
            [-0.4605, 0.7605],
            [0, 0.06],
            [0, 1],
        ])

    def step(self, action) -> dm_env.TimeStep:
        scaled_action = _scale_nested_action(action, self._action_spec, self._clip, self._clip_action_spec)
        return self._environment.step(scaled_action)

    def action_spec(self):
        return _convert_spec(self._environment.action_spec())


def _convert_spec(nested_spec):
    """Converts all bounded specs in nested spec to the canonical scale."""

    def _convert_single_spec(spec):
        """Converts a single spec to canonical if bounded."""
        if isinstance(spec, specs.BoundedArray):
            return spec.replace(
                minimum=-np.ones(spec.shape), maximum=np.ones(spec.shape)
            )
        else:
            return spec

    return tree.map_structure(_convert_single_spec, nested_spec)


def _scale_nested_action(nested_action, nested_spec, clip: bool, bound_array):
    """Converts a canonical nested action back to the given nested action spec."""

    def _scale_action(action: np.ndarray, spec: specs.Array):
        """Converts a single canonical action back to the given action spec."""
        if isinstance(spec, specs.BoundedArray):
            # Get scale and offset of output action spec.
            # scale = spec.maximum - spec.minimum
            scale = bound_array[:,1] - bound_array[:,0]
            # offset = spec.minimum
            offset = bound_array[:,0]

            # Maybe clip the action.
            if clip:
                action = np.clip(action, -1.0, 1.0)

            # Map action to [0, 1].
            action = 0.5 * (action + 1.0)

            # Map action to [spec.minimum, spec.maximum].
            action *= scale
            action += offset


        return action

    return tree.map_structure(_scale_action, nested_action, nested_spec)
