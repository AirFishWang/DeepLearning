# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     map_reduce
   Description :
   Author :        wangchun
   date：          18-9-5
-------------------------------------------------
   Change Activity:
                   18-9-5:
-------------------------------------------------
存在的问题： 当使用管道通信时，子进程会处于循环监听中， 当主进程不再发送消息到管道的时候， 子进程会因为recv()函数一直处于阻塞状态，无法退出

"""
import os, time
from multiprocessing import Process, Pipe, Event


class SonProcess(Process):
    def __init__(self, name, parent_side, child_pipe):
        super(SonProcess, self).__init__()
        self.name = name
        self.child_pipe = child_pipe
        self.parent_side = parent_side
        self.stopEvent = Event()

    def run(self):
        print "{} start!, pid = {}  parent pid = {}".format(self.name, os.getpid(), os.getppid())
        while True:
            image_list = self.child_pipe.recv()                 # process block
            if image_list == "stop":
                break
            time.sleep(1)
            image_list = [x*2 for x in image_list]
            self.child_pipe.send(image_list)
        print "{} end!, pid = {}  parent pid = {}".format(self.name, os.getpid(), os.getppid())


class FatherProcess():
    def __init__(self, count_process):
        self.count_process = count_process
        self.process_list = []
        for i in range(count_process):
            parent_side, child_side = Pipe()
            p = SonProcess("son_process_{}".format(i), parent_side, child_side)
            self.process_list.append(p)
            p.start()

    def process_func(self, image_list):
        # map
        for i in range(self.count_process):
            tmp_list = image_list[i::self.count_process]
            self.process_list[i].parent_side.send(tmp_list)

        recv_list = [self.process_list[i].parent_side.recv() for i in range(self.count_process)]  # wait for son process

        # reduce
        result = []
        max_length = max([len(x) for x in recv_list])
        for j in range(max_length):
            for i in range(len(recv_list)):
                if j < len(recv_list[i]):
                    result.append(recv_list[i][j])
        return result

    def stop_son_process(self):
        for son_process in self.process_list:
            son_process.parent_side.send("stop")

    def join_son_process(self):
        for son_process in self.process_list:
            son_process.join()


def multi_process_test():

    image_list = range(4)
    count_process = 4
    obj = FatherProcess(count_process)
    for i in range(2):
        print obj.process_func(image_list)
    obj.stop_son_process()
    obj.join_son_process()


def map_reduce_test():
    test_list = range(10)
    count_process = 3
    all_list = []
    for i in range(count_process):
        tmp_list = test_list[i::count_process]
        all_list.append(tmp_list)
        print tmp_list
    print all_list

    result = []
    max_length = max([len(x) for x in all_list])

    for j in range(max_length):
        for i in range(len(all_list)):
            if j < len(all_list[i]):
                result.append(all_list[i][j])

    print result


if __name__ == "__main__":
    print "main process start!, pid = {}".format(os.getpid())
    multi_process_test()
    print "main process end!, pid = {}".format(os.getpid())
