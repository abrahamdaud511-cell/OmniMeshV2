import unittest
from omnimesh.atsg import AdaptiveTrainingStabilityGovernor
from omnimesh.config import ModelConfig

class TestATSG(unittest.TestCase):
    def test_init(self):
        config = ModelConfig()
        atsg = AdaptiveTrainingStabilityGovernor(config)
        self.assertIsNotNone(atsg)
        # start/stop not tested in CI
