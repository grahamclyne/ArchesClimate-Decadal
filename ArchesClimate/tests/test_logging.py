"""Regressions for metric logging without a Trainer or GPU."""

import unittest
from types import SimpleNamespace

from ArchesClimate.model.base_module import BaseLightningModule


class MetricLoggingTests(unittest.TestCase):
    def test_successive_calls_only_log_current_metrics(self):
        calls = []
        model = SimpleNamespace(
            training=True,
            log=lambda name, value, **kwargs: calls.append((name, value)),
        )
        BaseLightningModule.mylog(model, loss=1.0)
        BaseLightningModule.mylog(model, rmse=2.0)
        self.assertEqual(calls, [("train_loss", 1.0), ("train_rmse", 2.0)])

    def test_different_models_do_not_share_metric_names(self):
        calls = []
        first = SimpleNamespace(training=True, log=lambda *args, **kwargs: None)
        second = SimpleNamespace(
            training=False,
            log=lambda name, value, **kwargs: calls.append((name, value)),
        )
        BaseLightningModule.mylog(first, loss=1.0)
        BaseLightningModule.mylog(second, crps=3.0)
        self.assertEqual(calls, [("val_crps", 3.0)])


if __name__ == "__main__":
    unittest.main()
