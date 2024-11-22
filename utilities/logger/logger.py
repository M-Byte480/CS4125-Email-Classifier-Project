# todo: review unused file

import string
import logging
class Logger:
    message : string

    def __init__(self, m : string):
        self.message = m

    def log(self):
        print("LOG: " + self.message)

    def log_warning(self):
        logging.warn(self.message)

    def log_error(self):
        logging.error(self.message)

class LoggerFactory:

    def __init__(self):
        print("Logger Factory is up!")

    def make_logger(self, message : string, flag : int):
        l = Logger(message)
        match flag:
            case 1:
                l.log()
            case 2:
                l.log_warning()
            case _:
                l.log_error()