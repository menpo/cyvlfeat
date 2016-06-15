from unittest import TestCase

class TestIsNumeric(TestCase):
    NUMERIC = [True, 1, -1, 1.0, 1+1j]
    NOT_NUMERIC = [object(), 'string', u'unicode', None]

    def test_is_numeric(self):
        for x in self.NUMERIC:
            for y in (x, [x], [x] * 2):
                for z in (y, np.array(y)):
                    self.assertTrue(is_numeric(z))
        for x in self.NOT_NUMERIC:
            for y in (x, [x], [x] * 2):
                for z in (y, np.array(y)):
                    self.assertFalse(is_numeric(z))
        for kind, dtypes in np.sctypes.items():
            if kind != 'others':
                for dtype in dtypes:
                    self.assertTrue(is_numeric(np.array([0], dtype=dtype)))
