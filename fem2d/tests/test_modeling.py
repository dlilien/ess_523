#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 dlilien <dlilien@berens>
#
# Distributed under terms of the MIT license.

"""
General tests for core/classes.py
"""

import unittest
from fem2d.core import eqnlist


class TestEquationList(unittest.TestCase):
    def test_eqn_list(self):
        eql=eqnlist()
        eql.setitem('a','a_val',0)
        eql.setitem('b','b_val')
        eql.setitem('c','c_val',before_all=True)
        eql.setitem('d','d_val',after_all=True)
        for key in sorted(eql.numbers):
            eql[key]
        self.assertEqual(eql['d'],'d_val')
        self.assertEqual(eql[1],'b_val')


if __name__=='__main__':
    unittest.main(buffer=True)




