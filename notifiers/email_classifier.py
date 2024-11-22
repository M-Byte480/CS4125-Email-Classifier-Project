from notifiers.observer.observer import Observer


class EmailClassifier:
    def __init__(self):
        self._observers = []

    def add_observer(self, observer: Observer) -> None:
        self._observers.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def classify_email(self, email) -> None:
        classification = "spam" if "spam" in email.lower() else "not spam"
        self.notify_observers(email, classification)

    def notify_observers(self, email, classification) -> None:
        for observer in self._observers:
            observer.update(email, classification)

