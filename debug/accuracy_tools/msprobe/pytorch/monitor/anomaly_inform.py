#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta

from msprobe.core.common.const import MonitorConst
from msprobe.pytorch.monitor.database import Database, ExceptionMessage
from msprobe.pytorch.monitor.utils import beijing_tz


# define class InformRegistry to get inform_sub_class
class AnomalyInformFactory:
    @staticmethod
    def create_informer(**kwargs):
        recipient = kwargs.get("recipient")
        if recipient == MonitorConst.DATABASE:
            return DatabaseInform(**kwargs)
        elif recipient == MonitorConst.EMAIL:
            return EmailInform(**kwargs)
        raise ValueError("Invalid recipient specified")


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
            self.current_time = datetime.now(tz=beijing_tz)
        if self.time == 0 or ((self.current_time - self.time) > timedelta(minutes=self.interval_time)):
            self.exception_message_list.append(exception_message)
            self.inform_fun(self.exception_message_list, job_id)
            self.exception_message_list = []
            self.time = datetime.now(tz=beijing_tz)
        elif (self.current_time - self.time) <= timedelta(minutes=self.interval_time):
            self.exception_message_list.append(exception_message)
            self.current_time = datetime.now(tz=beijing_tz)


class DatabaseInform(AnomalyInform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interval_time = 2
        self.database = Database(self.inform_args.get("connection_str", None))
        self.database.create_table()

    def inform_fun(self, exception_message_list, job_id):
        save_list = []
        for exception_message in exception_message_list:
            item = {
                'job_id': job_id,
                'message': exception_message,
                'create_time': datetime.now(tz=beijing_tz)
            }
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
            server.login(self.inform_args.get('send_email_username', None),
                         self.inform_args.get('send_email_password', None))
            server.sendmail(self.inform_args.get('send_email_address', None),
                            self.inform_args.get('receive_email_address', None), message.as_string())
