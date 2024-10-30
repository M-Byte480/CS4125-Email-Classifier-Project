from notification_entity.notify_behaviour import Notify_Behaviour


class Email(Notify_Behaviour):
    sender = "donotreply@jim-b.com"

    def __init__(self, subject: str, body: str, recipient: str):
        self.subject = subject
        self.body = body
        self.recipient = recipient

    def notify(self) -> None:
        print("Email sent!")

class SMS(Notify_Behaviour):
    def notify(self) -> None:
        print("SMS sent!")

class Push(Notify_Behaviour):
    def notify(self) -> None:
        print("Push notification sent!")

