"""Conditioning concatenation must address fields rather than batch entries."""

import unittest

import torch
from tensordict import TensorDict

from ArchesClimate.utils.tensordict_utils import tensordict_cat


class TensorDictConcatenationTests(unittest.TestCase):
    def test_surface_and_level_fields_keep_shapes_values_and_gradients(self):
        states = []
        for offset in [0.0, 10.0]:
            states.append(
                TensorDict(
                    {
                        "surface": torch.full(
                            (1, 2, 1, 3, 4), offset, requires_grad=True
                        ),
                        "level": torch.full(
                            (1, 1, 2, 3, 4), offset + 1, requires_grad=True
                        ),
                    },
                    batch_size=[1],
                )
            )
        result = tensordict_cat(states, dim=1)
        self.assertEqual(set(result.keys()), {"surface", "level"})
        for key in ["surface", "level"]:
            expected = torch.cat([state[key] for state in states], dim=1)
            torch.testing.assert_close(result[key], expected)
        sum(value.sum() for value in result.values()).backward()
        for state in states:
            for value in state.values():
                torch.testing.assert_close(value.grad, torch.ones_like(value))


if __name__ == "__main__":
    unittest.main()
