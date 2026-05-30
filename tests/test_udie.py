import unittest
from omnimesh.udie import UniversalDataIngestionEngine

class TestUDIE(unittest.TestCase):
    def setUp(self):
        self.udie = UniversalDataIngestionEngine()
    
    def test_text_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello world")
            path = f.name
        tokens = self.udie.ingest_file(path)
        self.assertIsNotNone(tokens)
        import os
        os.unlink(path)

if __name__ == '__main__':
    unittest.main()
