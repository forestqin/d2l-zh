import logging
import os
from logging.handlers import TimedRotatingFileHandler


def init_logger(level):
    if level == "DEBUG" or level == "debug":
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logger = logging.getLogger("logger")
    logger.setLevel(LEVEL)
    # datefmt = "%Y-%m-%d %H:%M:%S"
    datefmt = "%m-%d %H:%M:%S"
    # format_str = "[%(asctime)s]\t%(name)s\t%(filename)s[%(lineno)s]\t%(funcName)s\t%(levelname)s\t%(message)s"
    format_str = "[%(asctime)s][%(filename)s:%(funcName)s]%(levelname)s %(message)s"
    formatter = logging.Formatter(format_str, datefmt)

    if not os.path.exists("log"):
        os.makedirs("log")

    # handler all
    all_handler = TimedRotatingFileHandler('log/all.log',
                                           when='midnight',
                                           backupCount=7)
    all_handler.setFormatter(formatter)
    all_handler.setLevel(LEVEL)
    logger.addHandler(all_handler)

    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(LEVEL)
    logger.addHandler(ch)

    # handler error
    error_handler = TimedRotatingFileHandler('log/error.log',
                                             when='midnight',
                                             backupCount=7)
    error_handler.setFormatter(formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)

    return logger


# log = init_logger("house_price", level)


def main():
    logger = init_logger("debug")
    print(logger.getEffectiveLevel())
    logger.error("test-error")
    logger.info("test-info")
    logger.warning("test-warn")
    logger.debug("test debug")


if __name__ == '__main__':
    main()
