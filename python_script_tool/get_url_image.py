# -*- coding: utf-8 -*-
import cv2
import numpy as np
import requests
import traceback
import logging


# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.INFO)

# 创建一个handler，用于写入日志文件
fh = logging.FileHandler('test.log')
# fh.setLevel(logging.INFO)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(filename)s %(lineno)s: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
# logger.addHandler(fh)
logger.addHandler(ch)

def get_url_image():
    timeout = 0.001
    url = "https://developer.nvidia.com/sites/default/files/akamai/cuda/images/deeplearning/TRT3_Benchmark1.PNG"
    # url = "http://119.97.201.22:8080/GetHouseInfo.ashx?price=ABmTF%2BZJfC7jwz5yz77JLg=="
    try:
        url_image = requests.get(url, timeout=timeout)
    except Exception as ex:
        # traceback.print_exc()
        logger.info('callback failed: %s' % (ex))
        print "ex = ", ex
        print "traceback.format_exc() = {}".format(traceback.format_exc())
        exit()
    image = np.asarray(bytearray(url_image.content), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    cv2.imshow("image", image)
    cv2.waitKey()

if __name__ == "__main__":
    get_url_image()