# This source code is from https://github.com/open-mmlab/Amphion/blob/d3de90f02e7f1c7dbc24b04d52fe1b8ef438effe/preprocessors/Emilia/models/whisper_asr.py
# MIT

import logging
import time
import os


class Logger:
    """
    Logger class for managing logging operations.
    """

    _logger = None

    @classmethod
    def get_logger(cls, name=None):
        """
        Get the logger instance with the specified name. If it doesn't exist, create and cache it.

        Args:
            cls (type): The class type.
            name (str, optional): The name of the logger. Defaults to None, which uses the class name.

        Returns:
            logging.Logger: The logger instance.
        """
        if cls._logger is None:
            cls._logger = cls.init_logger(name)
        return cls._logger

    @classmethod
    def init_logger(cls, name=None):
        """
        Initialize the logger, including file and console logging.

        Args:
            cls (type): The class type.
            name (str, optional): The name of the logger. Defaults to None.

        Returns:
            logging.Logger: The initialized logger instance.
        """
        if name is None:
            name = "main"
            if "SELF_ID" in os.environ:
                name = name + "_ID" + os.environ["SELF_ID"]
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                name = name + "_GPU" + os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"Initialize logger for {name}")
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Libraries in the stack (speechbrain, lightning) call logging.basicConfig,
        # which installs a handler on the root logger. Propagating there as well
        # printed every record twice: once via our handlers below, once as
        # "LEVEL:name:message" from root. Re-running in a notebook cell also
        # stacked duplicate handlers on the cached logger.
        logger.propagate = False
        for existing in list(logger.handlers):
            logger.removeHandler(existing)

        # Add file handler to save logs to a file
        log_date = time.strftime("%Y-%m-%d", time.localtime())
        log_time = time.strftime("%H-%M-%S", time.localtime())
        os.makedirs(f"logs/{log_date}", exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(f"logs/{log_date}/{name}-{log_time}.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Create a custom log formatter to set specific log levels to color
        class ColorFormatter(logging.Formatter):
            """
            Custom log formatter to add color to specific log levels.
            """

            def format(self, record):
                """
                Format the log record with color based on log level.

                Args:
                    record (logging.LogRecord): The log record to format.

                Returns:
                    str: The formatted log message.
                """
                # Colour the formatted output, never record.msg: mutating the
                # record in place leaks ANSI codes into every other handler that
                # sees the same record, and nests them if it is formatted twice.
                if record.levelno >= logging.ERROR:
                    colour = "1;31"
                elif record.levelno >= logging.WARNING:
                    colour = "1;33"
                elif record.levelno >= logging.INFO:
                    colour = "1;34"
                else:
                    colour = "1;32"
                return f"\033[{colour}m{super().format(record)}\033[0m"

        color_formatter = ColorFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(color_formatter)
        logger.addHandler(ch)

        return logger


def time_logger(func):
    """
    Decorator to log the execution time of a function.

    Args:
        func (callable): The function whose execution time is to be logged.

    Returns:
        callable: The wrapper function that logs the execution time of the original function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        Logger.get_logger().debug(
            f"Function {func.__name__} took {end_time - start_time} seconds to execute"
        )
        return result

    return wrapper