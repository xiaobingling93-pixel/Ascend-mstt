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

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from pymysql.err import OperationalError

Base = declarative_base()


class ExceptionMessage(Base):
    __tablename__ = 'exception_message'

    id = Column(Integer, primary_key=True)
    job_id = Column(String(40), index=True, nullable=False)
    message = Column(String(255))
    create_time = Column(DateTime, nullable=False)

    def __repr__(self):
        return '<ExceptionMessage(job_id={}, message={})'.format(self.job_id, self.message)


class Database:
    def __init__(self, connection_str):
        """ connection_str = 'dialect+driver://username:password@host:port/database'"""

        self.engine = create_engine(connection_str, pool_recycle=25200)

    def get_session(self):
        ss = sessionmaker(bind=self.engine)
        return ss()

    def create_table(self):
        Base.metadata.create_all(self.engine, checkfirst=True)

    def find_by_job_id(self, job_id):
        session = self.get_session()
        try:
            exception_message = session.query(ExceptionMessage).filter(ExceptionMessage.job_id == job_id).all()
            return exception_message
        except OperationalError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def insert_batch(self, datas):
        session = self.get_session()
        try:
            session.bulk_save_objects(datas)
            session.commit()
        except OperationalError as e:
            session.rollback()
            raise e
        finally:
            session.close()
