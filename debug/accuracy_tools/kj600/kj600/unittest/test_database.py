import unittest
import uuid
from datetime import datetime
from unittest import TestCase

from sqlalchemy import inspect

from kj600.database import Database, ExceptionMessage


class TestDatabase(TestCase):
    def __init__(self, method_name: str):
        super(TestDatabase, self).__init__(method_name)
        self.db = Database('mysql+pymysql://username:password@host:port/database')

    def test_create_table(self):
        self.db.create_table()
        inspect_ = inspect(self.db.engine)
        table_names = inspect_.get_table_names()
        print(table_names)
        self.assertIn("exception_message", table_names)

    def test_insert_batch(self):
        self.db.create_table()
        job_id = str(uuid.uuid4())
        print(job_id)
        save_list = []
        exception_message_list = [
            '[93m> Rule AnomalyTurbulence reports anomaly signal in language_model.encoder.layers.0/1/input_zeros at step 1.[0m',
            '[93m> Rule AnomalyTurbulence reports anomaly signal in language_model.encoder.layers.0.input_norm.weight/0/exp_avg_min at step 2.[0m',
            '[93m> Rule AnomalyTurbulence reports anomaly signal in language_model.encoder.layers.0.input_norm.weight/1/exp_avg_min at step 2.[0m']
        for exception_message in exception_message_list:
            item = {'job_id': job_id, 'message': exception_message, 'create_time': datetime.now()}
            save_list.append(ExceptionMessage(**item))
        self.db.insert_batch(save_list)
        find_by_job_id = self.db.find_by_job_id(job_id)
        exception_messages = [item.message for item in find_by_job_id]
        self.assertEqual(exception_messages, exception_message_list)


if __name__ == '__main__':
    unittest.main()
