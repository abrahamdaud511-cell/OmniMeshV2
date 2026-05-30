import unittest
from omnimesh import OmniMeshV2, ModelConfig

class TestModel(unittest.TestCase):
    def test_init(self):
        config = ModelConfig()
        model = OmniMeshV2(config)
        self.assertIsNotNone(model)
