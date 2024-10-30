from abc import ABC, abstractmethod

class Notify_Behaviour(ABC):
    @abstractmethod
    def notify(self) -> None:
        """
        Sends out the notification
        :return: None
        """
        pass

