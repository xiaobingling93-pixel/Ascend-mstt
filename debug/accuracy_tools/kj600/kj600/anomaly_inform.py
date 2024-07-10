import smtplib
from email.mime.text import MIMEText
import sqlite3
from datetime import datetime, timedelta

from kj600.database import Database, ExceptionMessage


# define class InformRegistry to get inform_sub_class
class AnomalyInformFactory:
    @staticmethod
    def create_informer(**kwargs):
        if kwargs['recipient'] == "database":
            return DatabaseInform(**kwargs)
        elif kwargs['recipient'] == "email":
            return EmailInform(**kwargs)
        else:
            raise ValueError("Invaild recipient specified")

# define class AnomalyInform to inform with database or email
class AnomalyInform:
    def __init__(self, **kwargs):
        self.inform_args = kwargs
        self.exception_message_list = []
        self.time = 0
        self.current_time = 0

    def inform_fun(self, exception_message_list, job_id):
        pass

    def run(self, exception_message, job_id):
        if self.time != 0 and self.current_time == 0:
            self.current_time = datetime.now()
        if self.time == 0 or ((self.current_time - self.time) > timedelta(minutes=self.interval_time)):
            self.exception_message_list.append(exception_message)
            self.inform_fun(self.exception_message_list, job_id)
            self.exception_message_list = []
            self.time = datetime.now()
        elif (self.current_time - self.time) <= timedelta(minutes=self.interval_time):
            self.exception_message_list.append(exception_message)
            self.current_time = datetime.now()
        
class DatabaseInform(AnomalyInform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interval_time = 2
        self.database = Database(self.inform_args.get("connection_str", None))
        self.database.create_table()

    def inform_fun(self, exception_message_list, job_id):
        save_list = []
        for exception_message in exception_message_list:
            item = {'job_id': job_id, 'message': exception_message, 'create_time': datetime.now()}
            save_list.append(ExceptionMessage(**item))
        self.database.insert_batch(save_list)

class EmailInform(AnomalyInform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interval_time = 10

    def inform_fun(self, exception_message_list, job_id):
        subject = "Exception Detected in Your Program"
        text = f"{len(exception_message_list)} exception was detected in your program:\n\n"
        for exception_message in exception_message_list:
            text += f"{job_id}: {exception_message}\n"
        message = MIMEText(text, "plain")
        message["Subject"] = subject
        message["From"] = self.inform_args.get('send_email_address', None)
        message["To"] = self.inform_args.get('receive_email_address', None)

        with smtplib.SMTP(self.inform_args.get('smtp_server', None), self.inform_args.get('smtp_port', 587)) as server:
            server.starttls()
            server.login(self.inform_args.get('send_email_username', None), self.inform_args.get('send_email_password', None))
            server.sendmail(self.inform_args.get('send_email_address', None), 
                            self.inform_args.get('receive_email_address', None), message.as_string())
