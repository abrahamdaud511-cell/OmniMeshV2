import unittest
from omnimesh.safety import ConstitutionalSafetyRouterV2

class TestSafety(unittest.TestCase):
    def test_safe_output(self):
        router = ConstitutionalSafetyRouterV2()
        output = "Hello, how are you?"
        revised = router.check_and_revise(output)
        self.assertEqual(revised, output)
    
    def test_unsafe_code(self):
        router = ConstitutionalSafetyRouterV2()
        output = "```python\nimport os; os.system('rm -rf /')\n```"
        revised = router.check_and_revise(output)
        self.assertIn("SAFETY WARNING", revised)
