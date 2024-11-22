from colorama import Fore

from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.decorators.colour_decorator import ColourDecorator


class BaseWarningLogger(AbstractLogger):
    def __init__(self):
        super().__init__()
        self.base_message = "[WARNING]"

    def log(self, message):
        print(self.base_message, message)

class WarningLogger(BaseWarningLogger):
    def __init__(self):
        super().__init__()

    def log(self, message):
        decorated_logger = ColourDecorator(super(), Fore.YELLOW)
        decorated_logger.log(message)