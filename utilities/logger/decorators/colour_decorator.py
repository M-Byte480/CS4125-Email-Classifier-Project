from colorama import Style

from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.decorators.logger_decorator_interface import ILoggerDecorator


class ColourDecorator(ILoggerDecorator):
    def __init__(self, logger: AbstractLogger, colour_code: str):
        super().__init__(logger)
        self.colour_code = colour_code
        self.reset_code = Style.RESET_ALL

    def log(self, message):
        colored_message = f"{self.colour_code}{message}{self.reset_code}"
        self.logger.log(colored_message)