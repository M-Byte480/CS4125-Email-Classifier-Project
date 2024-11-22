from abc import ABC, abstractmethod

class AbstractLogger(ABC):
    __slots__ = ["base_message"]

    def __init__(self):
        self.base_message = "[BASE LOG]"
        ...

    @abstractmethod
    def log(self, message: str):
        pass
