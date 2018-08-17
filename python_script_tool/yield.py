# -*- coding: utf-8 -*-
import random


def test_yield():
    def gen():
        while True:
            list = [1, 2, 3, 4]
            random.shuffle(list)
            for x in list:
                yield x
    f = gen()
    for i in range(10):
        print f.next()


if __name__ == "__main__":
    test_yield()
