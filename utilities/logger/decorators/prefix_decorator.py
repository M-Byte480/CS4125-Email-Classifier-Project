from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.decorators.logger_decorator_interface import ILoggerDecorator


class PrefixLogger(ILoggerDecorator):
    def __init__(self, logger: AbstractLogger, prefix: str = ""):
        super().__init__(logger)
        self.prefix = prefix


    def log(self, message = ""):
        self.logger.log(f"[{self.prefix}] {message}")
