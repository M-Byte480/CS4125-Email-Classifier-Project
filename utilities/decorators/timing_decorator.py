import time
from functools import wraps

from utilities.logger.concrete_logger.info_logger import InfoLogger
from utilities.logger.decorators.prefix_decorator import PrefixLogger


class TimingDecorator:
    def __init__(self, info_logger=None) -> None:
        self.info_logger = info_logger or InfoLogger()
        self.info_logger = PrefixLogger(self.info_logger, "TimingDecorator")

    def __call__(self, method):
        @wraps(method)
        def timing_decorator(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            # Log the execution time
            self.info_logger.log(f"Execution time for {method.__name__}: {execution_time:.4f} seconds")
            return result
        return timing_decorator