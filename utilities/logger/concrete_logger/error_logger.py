from utilities.logger.abstract_logger import AbstractLogger
from utilities.logger.decorators.background_decorator import BackgroundDecorator
from utilities.logger.decorators.bold_decorator import BoldDecorator
from utilities.logger.decorators.colour_decorator import ColourDecorator
from colorama import Fore, Back

from utilities.logger.decorators.italics_decorator import ItalicsDecorator


class BaseErrorLogger(AbstractLogger):

    def __init__(self):
        super().__init__()
        self.base_message = "[ERROR]"


    def log(self, message):
        print(self.base_message, message)

class ErrorLogger(BaseErrorLogger):
    def __init__(self):
        super().__init__()


    def log(self, message):
        decorated_logger = ColourDecorator(super(), Fore.RED)
        decorated_logger = BackgroundDecorator(decorated_logger, Back.BLACK)
        decorated_logger = BoldDecorator(decorated_logger)
        decorated_logger = ItalicsDecorator(decorated_logger)
        decorated_logger.log(message)