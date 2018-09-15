# -*- coding: utf-8 -*-
import logging

"""
logging.basicConfig < handler.setLevel < logger.setLevel
1.脚本中没有配置logger.setLevel会使用handler.setLevel
2.脚本中没有配置logger.setLevel和handler.setLevel会使用logging.basicConfig中的Level等级（默认WARNING）

# 日志级别： debug < info < warning < error < critical
"""


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
formatter = logging.Formatter('[%(asctime)s][%(filename)s][line: %(lineno)d][%(levelname)s]: %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)
