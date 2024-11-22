# todo: review unused file

import string

class Email:
    id : int
    ticket_id : int
    sender : string
    recipient : string
    subject : string
    body_text : string
    classification : string

    def __init__(self, i, t, sen, r, sub, bt, c):
        self.id = i
        self.ticket_id = t
        self.sender = sen
        self.recipient = r
        self.subject = sub
        self.body_text = bt
        self.classification = c

    def get_id(self):
        return self.id

    def get_ticket_id(self):
        return self.ticket_id

    def get_sender(self):
        return self.sender

    def get_recipient(self):
        return self.recipient

    def get_subject(self):
        return self.subject

    def get_body_text(self):
        return self.body_text

    def get_classification(self):
        return self.classification

    def set_id(self, new_id):
        self.id = new_id

    def set_ticket_id(self, t_id):
        self.ticket_id = t_id

    def set_sender(self, s):
        self.sender = s

    def set_recipient(self, r):
        self.recipient = r

    def set_subject(self, s):
        self.subject = s

    def set_body_text(self, bt):
        self.body_text = bt

    def setClassification(self, c):
        self.classification = c

    def notifyEm(self, n):
        n.send_email(self)

class Notifier:

    sender_address : string

    def __init__(self, sender_address):
        self.sender_address = sender_address

    def send_email(self, msg : string):
        print("Dear " + self.sender_address + ", " + msg)

    def send_email(self, e: Email):
        self.sender_address = e.get_sender()
        print("Dear " + e.get_recipient() + ", " + e.get_body_text())