#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import logging

def get_logger(name=None, verbose=False, handler=logging.StreamHandler()):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(fmt="[%(levelname).1s %(asctime)s %(funcName)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler = handler
    log_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    log_handler.setFormatter(formatter)
    # logger.handlers = []
    logger.addHandler(log_handler)
    return logger


if __name__ == "__main__":
    message = "test log"
    logger = get_logger()
    logger.info(message)
    print("finish test")
