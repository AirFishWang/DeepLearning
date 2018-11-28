# -*- coding: utf-8 -*-
import time
from log import logger
global logger


if __name__ == "__main__":
    for i in range(8):
        logger.info("this is a info message")
        logger.debug("this is a debug message")
        logger.warning("this is a warning message")
        logger.error("this is a error message")
        time.sleep(15)