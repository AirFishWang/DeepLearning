# -*- coding: utf-8 -*-
import random


def gen():
    while True:
        list = [1, 2, 3, 4]
        random.shuffle(list)
        for x in list:
            yield x


def test_yield():
    f = gen()
    for i in range(10):
        print f.next()

    for x in f:
        print x


if __name__ == "__main__":
    test_yield()

