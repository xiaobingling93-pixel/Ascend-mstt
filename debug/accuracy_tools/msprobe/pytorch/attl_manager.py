from msprobe.core.common.runtime import Runtime
from msprobe.core.common.utils import Const
from msprobe.pytorch.api_accuracy_checker.common.utils import ApiData
from msprobe.pytorch.common.log import logger


class ATTLManager:
    def __init__(self, config):
        self.config = config
        self.attl = None

    def attl_init(self):
        if self.config.online_run_ut:
            from msprobe.pytorch.api_accuracy_checker.tensor_transport_layer.attl import ATTLConfig, ATTL
            attl_config = ATTLConfig(is_benchmark_device=False,
                                     connect_ip=self.config.host,
                                     connect_port=self.config.port,
                                     nfs_path=self.config.nfs_path,
                                     tls_path=self.config.tls_path)
            need_dump = len(self.config.rank) == 0 or self.current_rank in self.config.rank
            self.attl = ATTL('npu', attl_config, need_dump=need_dump)
            if self.config.nfs_path:
                self.attl.upload("start")

    def attl_send(self, name, args, kwargs, output):
        api_data = ApiData(
                    name[:-len(Const.FORWARD_NAME_SUFFIX)],
                    args,
                    kwargs,
                    output,
                    Runtime.current_iter,
                    Runtime.current_rank
                )
        logger.info(f"tools is dumping api: {api_data.name}, rank: {self.current_rank}")
        api_type, _, _ = api_data.name.split(Const.SEP)
        if api_type in [Const.DISTRIBUTED]:
            logger.info(f"api {api_data.name} is not supported, skip")
            return
        if self.config.nfs_path:
            self.attl.upload(api_data)
        else:
            self.attl.send(api_data)

    def attl_stop(self):
        if self.config.nfs_path:
            self.attl.upload("end")
        elif self.attl.socket_manager is not None:
            logger.info(f"pid: {os.getpid()} finished, start sends STOP signal.")
            self.attl.socket_manager.send_stop_signal()