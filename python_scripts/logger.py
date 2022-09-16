import logging as lg
import sys


def custom_logger(name):
    formatter = lg.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                             datefmt='%d-%m-%Y %H:%M:%S')
    handler = lg.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = lg.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = lg.getLogger(name)
    logger.setLevel(lg.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
