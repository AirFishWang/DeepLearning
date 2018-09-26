# -*- coding: utf-8 -*-
"""
    类似于结构体
"""

from collections import namedtuple


if __name__ == "__main__":
    coordinate = namedtuple('Coordinate', ['x', 'y'])
    co = coordinate(10, 20)
    print co.x, co.y
    print co[0], co[1]
    co = coordinate._make([100, 200])
    print co.x, co.y
    co = co._replace(x=30)
    print co.x, co.y
