class Notification:
    def __init__(self, medium: Medium):
        self.medium = medium

    def notify(self) -> None:
        self.medium.notify()