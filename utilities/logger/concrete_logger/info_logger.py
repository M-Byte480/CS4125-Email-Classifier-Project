from colorama import Fore

from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.decorators.colour_decorator import ColourDecorator


class BaseInfoLogger(AbstractLogger):
    def __init__(self):
        super().__init__()
        self.base_message = "[INFO]"

    def log(self, message):
        print(self.base_message, message)

class InfoLogger(BaseInfoLogger):
    def __init__(self):
        super().__init__()


    def log(self, message):
        coloured_logger = ColourDecorator(super(), Fore.LIGHTCYAN_EX)
        coloured_logger.log(message)