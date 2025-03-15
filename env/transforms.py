from typing import Sequence

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import TensorSpec, Unbounded, Composite
from torchrl.envs import Transform


class AttachZeroDimTensor(Transform):

    def __init__(self, in_keys: Sequence[NestedKey] = None, out_keys: Sequence[NestedKey] | None = None,
                 in_keys_inv: Sequence[NestedKey] | None = None, out_keys_inv: Sequence[NestedKey] | None = None,
                 attach_keys: Sequence[NestedKey] = None):
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        self.attach_keys = attach_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return super().forward(tensordict)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return super()._call(tensordict_reset)

    def _apply_transform(self, obs: TensorDict) -> Tensor:
        values = [obs.get(key) for key in self.attach_keys]
        shapes = [value.shape for value in values]
        index = shapes.index(min(shapes))
        expanding_value = values.pop(index)
        max_dim = max(shapes)[0]
        expanded_tensor = torch.expand_copy(expanding_value, (max_dim, 1))
        values.append(expanded_tensor)
        concated = torch.concat(values, dim=1)
        return concated

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return Composite(observation=Unbounded(shape=(15, 13), dtype=torch.float32))

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        data, budget = state.split([12,1], dim=1)
        return TensorDict({"data": data, "budget": budget})
