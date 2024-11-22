from abc import ABC, abstractmethod

from structs.objects import Email


class Observer(ABC):
    @abstractmethod
    def update(self, email: Email, classification):
        pass