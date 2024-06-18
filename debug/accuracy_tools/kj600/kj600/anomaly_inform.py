import smtplib
from email.mime.text import MIMEText
import sqlite3
from datetime import datetime, timedelta

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

    def inform_fun(self, exception_message_list):
        pass

    def run(self, exception_message):
        if self.time != 0 and self.current_time == 0:
            self.current_time = datetime.now()
        if self.time == 0 or ((self.current_time - self.time) > timedelta(minutes=self.interval_time)):
            self.exception_message_list.append(exception_message)
            self.inform_fun(self.exception_message_list)
            self.exception_message_list = []
            self.time = datetime.now()
        elif (self.current_time - self.time) <= timedelta(minutes=self.interval_time):
            self.exception_message_list.append(exception_message)
            self.current_time = datetime.now()
        
class DatabaseInform(AnomalyInform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interval_time = 2

    def inform_fun(self, exception_message_list):
        with sqlite3.connect(self.inform_args['connection_str']) as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS exceptions(
                                id INTEGER PRIMARY KEY,
                                message TEXT
                            )''')
            now_time = datetime.now()
            for exception_message in exception_message_list:
                exception_message = f"Current time is :{now_time}" + exception_message
                cursor.execute("INSERT INTO exceptions (message) VALUES (?)",(exception_message,))

class EmailInform(AnomalyInform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interval_time = 10
    
    def inform_fun(self, exception_message_list):
        subject = "Exception Detected in Your Program"
        text = f"{len(exception_message_list)} exception was detected in your program:\n\n"
        for exception_message in exception_message_list:
            text += exception_message + '\n'
        message = MIMEText(text, "plain")
        message["Subject"] = subject
        message["From"] = self.inform_args['email']
        message["To"] = self.inform_args['email']

        with smtplib.SMTP(self.inform_args['smtp_server_name'], self.inform_args.get('smtp_number', 587)) as server:
            server.starttls()
            server.login(self.inform_args['id'], self.inform_args['password'])
            server.sendmail(self.inform_args['email'], self.inform_args['email'], message.as_string())    
