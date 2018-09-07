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
import os, sys, time, psutil, signal
from multiprocessing import Process, Pipe


class SonProcess(Process):
    def __init__(self, name, parent_side, child_pipe):
        super(SonProcess, self).__init__()
        self.name = name
        self.child_pipe = child_pipe
        self.parent_side = parent_side

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


def get_pid(pid_log):
    result = {'father': [], 'son': []}
    if not os.path.exists(pid_log):
        return result
    with open(pid_log, 'r') as fread:
        lines = fread.readlines()
    lines = [x.strip() for x in lines if len(x.strip()) > 0]
    for x in lines:
        name, pid = x.split()
        result[name].append(int(pid))
    return result


def is_running(pid):
    if os.path.exists('/proc/%d' % pid):
        return True
    return False


def check_peocess_status(pid_status):
    father_pid = pid_status['father'][0]
    son_pid_list = pid_status['son']
    if is_running(father_pid):
        print "father process {} is running normally!".format(father_pid)
    else:
        print "father process {} is not running!".format(father_pid)

    for son_pid in son_pid_list:
        if is_running(son_pid):
            print "son process {} is running normally".format(son_pid)
        else:
            print "son process {} is not running!".format(son_pid)
        if psutil.Process(son_pid).ppid() != father_pid:
            print "son process {} is not the child process of father process {} actually!".format(son_pid, father_pid)


def multi_process_test(phase, pid_log):
    pid_status = get_pid(pid_log)
    print "pid_status = {}".format(pid_status)
    if phase == "status":
        if pid_status['father'] == []:
            print "There are no process is running"
        else:
            print "father pid = {}".format(pid_status['father'][0])
            for x in pid_status['son']:
                print "son pid = {}".format(x)
            check_peocess_status(pid_status)

    elif phase == "start":
        if pid_status['father'] != [] and is_running(pid_status['father'][0]):
            print "process have exists!"
            exit()
        image_list = range(4)
        count_process = 4
        obj = FatherProcess(count_process)
        with open(pid_log, 'w') as fwrite:
            fwrite.write('father {}\n'.format(os.getpid()))
            p = psutil.Process(os.getpid())
            for x in p.children():
                fwrite.write("son {}\n".format(x.pid))
        for i in range(2):
            print obj.process_func(image_list)
        # obj.stop_son_process()
        obj.join_son_process()
    elif phase == "stop":
        if pid_status['father'] == []:
            print "There are no process is running"
            exit()

        try:
            # kill son process
            for x in pid_status['son']:
                # make sure the process would be killed is the son of father process
                if psutil.Process(x).ppid() == pid_status['father'][0] and is_running(x):
                    os.kill(x, signal.SIGKILL)
                    print "kill son process {} success!".format(x)

            # kill father process
            if is_running(pid_status['father'][0]):
                os.kill(pid_status['father'][0], signal.SIGKILL)
                print "kill father process {} success!".format(pid_status['father'][0])
        except:
            print "kill process failed!"

        if os.path.exists(pid_log):
            os.remove(pid_log)
    else:
        print "parameter error!"


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
    help_msg = 'Usage: python {} <start|stop|status>'.format(sys.argv[0])
    if len(sys.argv) != 2:
        print help_msg
        sys.exit(1)

    pid_log = "./pid_log.txt"

    phase = sys.argv[1]
    print "phase = {}".format(phase)
    multi_process_test(phase, pid_log)

