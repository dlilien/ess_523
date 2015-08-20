#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright David Lilien dlilien90@gmail.com
#
# Distributed under terms of the MIT license.

"""
General tests for core/meshes.py
"""

import unittest
from fem2d.core import eqnlist
from nose.plugins.attrib import attr

@attr(slow=False)
class TestEquationList(unittest.TestCase):

    def test_eqn_list(self):
        eql = eqnlist()
        eql.setitem('a', 'a_val', 0)
        eql.setitem('b', 'b_val')
        eql.setitem('c', 'c_val', execute='before_all')
        eql.setitem('d', 'd_val', execute='after_all')
        for key in sorted(eql.numbers):
            eql[key]
        self.assertEqual(eql['d'], 'd_val')
        self.assertEqual(eql[1], 'b_val')


if __name__ == '__main__':
    unittest.main(buffer=True)
