"""Tests for the `lvarray` extension module"""

import unittest

import testPythonListOfStrings as lvarray


class ListOfStringsTests(unittest.TestCase):
    """Tests for the `lvarray` extension module"""

    def test_set(self):
        for setter in (lvarray.setarray, lvarray.setvector):
            for initializer in ("foobar", "barfoo", "hello world!"):
                strlist = setter(initializer)
                self.assertEqual(len(strlist), lvarray.ARR_SIZE)
                self.assertEqual([initializer for _ in range(lvarray.ARR_SIZE)], strlist)

    def test_get(self):
        lvarray.setarray("foobar")
        lvarray.setvector("barfoo")
        for getter in (lvarray.getarray, lvarray.getvector):
            self.assertEqual(getter(), getter())
            self.assertFalse(getter() is getter())


if __name__ == "__main__":
    unittest.main()
