# -*- coding: utf-8 -*-
import random


def gen():
    list = [1, 2, 3, 4]
    while True:
        # random.shuffle(list)
        # for x in list:
        #     yield x
        yield list[0]
        random.shuffle(list)
        print list


def yield_test():
    f = gen()
    for i in range(10):
        print f.next()

    # for x in f:
    #     print x


if __name__ == "__main__":
    yield_test()

