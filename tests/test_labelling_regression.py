import unittest

from tests.test_labelling_regression_support import ExistingRouteRegression


class F15ExistingRouteRegression(unittest.TestCase):
    def test_finite_inventory_and_seeded_response_state_equivalence(self):
        result = ExistingRouteRegression.run()
        self.assertEqual(result["unexpected_differences"], [])
        self.assertEqual(result["missing_cases"], [])


if __name__ == "__main__":
    unittest.main()
