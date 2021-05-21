import unittest
import axionpy.units as u
from axionpy import axis

class TestAxis(unittest.TestCase):
    def test_is_quantity(self):
        self.assertTrue(axis._is_quantity(1.0*u.s, "time"))
        self.assertTrue(axis._is_quantity(1.0*u.GeV, "energy"))
        self.assertFalse(axis._is_quantity(1.0*u.GeV, "time"))
        self.assertFalse(axis._is_quantity(1.0, "time"))

if __name__=='__main__':
    unittest.main()
