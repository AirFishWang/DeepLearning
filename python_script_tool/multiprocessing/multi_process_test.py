# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     multi_process_test
   Description :
   Author :        wangchun
   date：          18-9-5
-------------------------------------------------
   Change Activity:
                   18-9-5:
-------------------------------------------------
"""
import os
import time
import random
from multiprocessing import Process


def test_1():
    # 开进程的方法一:
    def piao(name):
        print '{} piaoing, working on process {}'.format(name, os.getpid())
        time.sleep(random.randrange(1, 5))
        print '{} piao end'.format(name)

    p1 = Process(target=piao, args=('egon',))  # 必须加,号
    p2 = Process(target=piao, args=('alex',))
    p3 = Process(target=piao, args=('wupeqi',))
    p4 = Process(target=piao, args=('yuanhao',))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    print('主线程')


def test_2():
    # 开进程的方法二:
    import time
    import random
    from multiprocessing import Process

    class Piao(Process):
        def __init__(self, name):
            super(Piao, self).__init__()
            self.name = name

        def run(self):
            print '{} piaoing, working on process {}'.format(self.name, os.getpid())
            time.sleep(random.randrange(1, 3))
            print '{} piao end'.format(self.name)

    p1 = Piao('egon')
    p2 = Piao('alex')
    p3 = Piao('wupeiqi')
    p4 = Piao('yuanhao')

    p1.start()  # start会自动调用run
    p2.start()
    p3.start()
    p4.start()
    print('主线程')

    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()

if __name__ == "__main__":
    print 'main process id = {} start'.format(os.getpid())
    test_2()
    print 'main process id = {} end'.format(os.getpid())