from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.logger_decorator_interface import ILoggerDecorator


class IndentationDecorator(ILoggerDecorator):
    message = ""
    def __init__(self, logger: AbstractLogger, message = ""):
        super().__init__(logger)
        if isinstance(logger, IndentationDecorator):
            self.indent_level = logger.indent_level + 1
        else:
            self.indent_level = 1

        self.message = self.message + message

    def log(self, message = ""):
        indentations = "    " * self.indent_level
        self.logger.log(f"{indentations}{message}")
