from colorama import Style

from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.decorators.logger_decorator_interface import ILoggerDecorator


class BackgroundDecorator(ILoggerDecorator):
    def __init__(self, logger: AbstractLogger, colour_code: str):
        super().__init__(logger)
        self.background_colour = colour_code
        self.reset_code = Style.RESET_ALL

    def log(self, message):
        colored_message = f"{self.background_colour}{message}{self.reset_code}"
        self.logger.log(colored_message)