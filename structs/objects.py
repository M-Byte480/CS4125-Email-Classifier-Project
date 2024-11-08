class Email:
    subject: str
    content: str

    def __init__(self, subject, content):
        self.subject = subject
        self.content = content