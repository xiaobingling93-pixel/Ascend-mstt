import uuid
import unittest

from kj600.anomaly_inform import AnomalyInformFactory


class TestAnomalyInform(unittest.TestCase):
    def test_database_inform(self):
        inform_args = {"inform": {"recipient": "database", "connection_str": "mysql+pymysql://username:password@host:port/database"}}
        anomaly_inform = AnomalyInformFactory.create_informer(**inform_args["inform"])
        exception_message = '\x1b[93m> Rule AnomalyTurbulence reports anomaly signal in language_model.encoder.layers.0.self_attention.query_key_value.weight/0/exp_avg_sq_min at step 49.\x1b[0m'
        job_id = str(uuid.uuid4())
        anomaly_inform.run(exception_message, job_id)

    def test_email_inform(self):
        inform_args = {"inform": {"recipient": "email", "send_email_address": "test@huawei.com", "receive_email_address": "test@huawei.com",
                                  "send_email_username": "foo", "send_email_password": "********",
                                  "smtp_server": "smtpscn.huawei.com", "smtp_port": "587"}}
        anomaly_inform = AnomalyInformFactory.create_informer(**inform_args["inform"])
        exception_message = '\x1b[93m> Rule AnomalyTurbulence reports anomaly signal in language_model.encoder.layers.0.self_attention.query_key_value.weight/0/exp_avg_sq_min at step 49.\x1b[0m'
        job_id = str(uuid.uuid4())
        anomaly_inform.run(exception_message, job_id)


if __name__ == "__main__":
    unittest.main()
