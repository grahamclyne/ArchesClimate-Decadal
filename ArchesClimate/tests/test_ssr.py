"""SSR must account for the number of members in the verifying mean."""

import unittest

import torch

from ArchesClimate.metrics.ensemble_metrics import EnsembleMetrics


class EnsembleMeanSSRTests(unittest.TestCase):
    def score(self, reference_members):
        # Five member deviations have sample variance exactly one. The
        # prescribed error is sqrt(1/5 + 1/K), the exchangeable-mean variance.
        members = torch.arange(-2, 3, dtype=torch.float32) / (2.5**0.5)
        predictions = members[None, :, None, None, None].expand(1, 5, 1, 4, 4)
        targets = torch.full((1, 1, 4, 4), (1 / 5 + 1 / reference_members) ** 0.5)
        metric = EnsembleMetrics(data_shape=(1,), reference_members=reference_members)
        metric.update(targets, predictions)
        return metric.compute()["spskr"]

    def test_five_member_reference_has_unit_balance(self):
        self.assertAlmostEqual(self.score(5).item(), 1.0, places=6)

    def test_different_reference_size_has_unit_balance(self):
        self.assertAlmostEqual(self.score(10).item(), 1.0, places=6)

    def test_single_realization_convention_is_preserved(self):
        self.assertAlmostEqual(self.score(1).item(), 1.0, places=6)

    def test_invalid_reference_size_is_rejected(self):
        with self.assertRaises(ValueError):
            EnsembleMetrics(data_shape=(1,), reference_members=0)


if __name__ == "__main__":
    unittest.main()
