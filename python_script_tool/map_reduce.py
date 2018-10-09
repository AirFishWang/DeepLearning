# -*- coding: utf-8 -*-

def map_reduce_test():
    list = range(1)
    work = 2

    # map
    map_list = []                                     # if the list length smaller than work, the tmp_list equal []
    for i in range(work):
        tmp_list = list[i::work]
        print "tmp_list = {}".format(tmp_list)
        map_list.append(tmp_list)

    # process
    for index, list in enumerate(map_list):
        map_list[index] = [x*2 for x in map_list[index]]
        print "map_list[{}] = {}".format(index, map_list[index])


    # reduce
    reduce_list = []
    max_length = max([len(x) for x in map_list])
    for j in range(max_length):
        for i in range(len(map_list)):
            if j < len(map_list[i]):
                reduce_list.append(map_list[i][j])
    print "reduce_list = {}".format(reduce_list)

if __name__ == "__main__":
    map_reduce_test()