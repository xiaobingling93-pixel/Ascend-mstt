# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
#
# Copyright(c) 2023 Huawei Technologies.
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
#
# Modifications: Add visualization of PyTorch Ascend profiling.
# --------------------------------------------------------------------------
import bisect
import os
import sys
from collections import defaultdict
from typing import List, Tuple

from .. import consts, io, utils
from ..multiprocessing import Process, Queue
from ..run import Run, RunProfile
from .data import DistributedRunProfileData, RunProfileData
from .node import CommunicationNode
from .run_generator import DistributedRunGenerator, RunGenerator

logger = utils.get_logger()


class RunLoader(object):
    def __init__(self, name, run_dir, caches: io.Cache, device_target="GPU"):
        self.run_name = name
        self.run_dir = run_dir
        self.caches = caches
        self.queue = Queue()
        self.device_target = device_target

    def load(self):
        workers = []
        spans_by_workers = defaultdict(list)
        if self.device_target == 'Ascend':
            for path in io.listdir(self.run_dir):
                if io.isdir(io.join(self.run_dir, path)) and utils.is_worker_span_dir(path) and io.isdir(
                        io.join(self.run_dir, path, 'ASCEND_PROFILER_OUTPUT')):
                    data_path = io.join(self.run_dir, path, 'ASCEND_PROFILER_OUTPUT')
                    for file in io.listdir(data_path):
                        if utils.is_npu_trace_path(file) or str(file) in (
                                'kernel_details.csv', 'memory_record.csv', 'operator_memory.csv',
                                'operator_details.csv'):
                            match = consts.WORKER_SPAN_PATTERN.match(path)
                            worker = match.group(1)
                            span = match.group(2)
                            if span is not None:
                                bisect.insort(spans_by_workers[worker], span)
                            workers.append((worker, span, io.join(path, 'ASCEND_PROFILER_OUTPUT')))
                            break
        else:
            for path in io.listdir(self.run_dir):
                if io.isdir(io.join(self.run_dir, path)):
                    continue
                match = consts.WORKER_PATTERN.match(path)
                if not match:
                    continue

                worker = match.group(1)
                span = match.group(2)
                if span is not None:
                    # remove the starting dot (.)
                    span = span[1:]
                    bisect.insort(spans_by_workers[worker], span)

                workers.append((worker, span, path))

        span_index_map = {}
        for worker, span_array in spans_by_workers.items():
            for i, span in enumerate(span_array, 1):
                span_index_map[(worker, span)] = i

        for worker, span, path in workers:
            # convert the span timestamp to the index.
            span_index = None if span is None else span_index_map[(worker, span)]
            p = Process(target=self._process_data, args=(worker, span, span_index, path))
            p.start()
        logger.info('started all processing')

        distributed_run = Run(self.run_name, self.run_dir, self.device_target)
        run = Run(self.run_name, self.run_dir, self.device_target)
        num_items = len(workers)
        while num_items > 0:
            item: Tuple[RunProfile, DistributedRunProfileData] = self.queue.get()
            num_items -= 1
            r, d = item
            if r or d:
                logger.debug('Loaded profile via mp.Queue')
            if r is not None:
                run.add_profile(r)
            if d is not None:
                distributed_run.add_profile(d)

        distributed_profiles = self._process_spans(distributed_run)
        for d in distributed_profiles:
            if d is not None:
                run.add_profile(d)

        # for no daemon process, no need to join them since it will automatically join
        return run

    def _process_data(self, worker, span_name, span, path):
        import absl.logging
        absl.logging.use_absl_handler()

        try:
            logger.debug('Parse trace, run_dir=%s, data_dir=%s', self.run_dir, path)
            local_file = self.caches.get_remote_cache(io.join(self.run_dir, path))
            if self.device_target == 'Ascend':
                data = RunProfileData.parse_npu(worker, span, local_file, self.caches.cache_dir)
            else:
                data = RunProfileData.parse_gpu(worker, span, local_file, self.caches.cache_dir)
                if not data:
                    self.queue.put((None, None))
                    logger.debug('finishing process data')
                    return
            if data.trace_file_path != local_file:
                self.caches.add_file(local_file, data.trace_file_path)

            generator = RunGenerator(worker, span, data, self.device_target)
            profile = generator.generate_run_profile()
            if self.device_target == 'Ascend':
                data.step_to_overlap = profile.step_to_overlap
                data.step_to_wait = profile.step_to_wait
                data.comm_op = profile.comm_op
            dist_data = DistributedRunProfileData(data)

            logger.debug('Sending back profile via mp.Queue')
            self.queue.put((profile, dist_data))
        except KeyboardInterrupt:
            logger.warning('tb_plugin receive keyboard interrupt signal, process %d will exit' % (os.getpid()))
            sys.exit(1)
        except Exception as ex:
            if self.device_target == 'Ascend':
                worker_name = f'{worker}_{span_name}_ascend_pt'
            else:
                worker_name = worker
            logger.warning('Failed to parse profile data for Run %s on %s. Exception=%s',
                           self.run_name, worker_name, ex, exc_info=True)
            self.queue.put((None, None))
        logger.debug('finishing process data')

    def _process_spans(self, distributed_run: Run):
        spans = distributed_run.get_spans()
        if spans is None:
            return [self._process_distributed_profiles(distributed_run.get_profiles(), None)]
        else:
            span_profiles = []
            for span in spans:
                profiles = distributed_run.get_profiles(span=span)
                p = self._process_distributed_profiles(profiles, span)
                if p is not None:
                    span_profiles.append(p)
            return span_profiles

    def _process_distributed_profiles(self, profiles: List[DistributedRunProfileData], span):
        if self.device_target != 'Ascend':
            return self._gpu_distributed(profiles, span)
        else:
            for data in profiles:
                if not data.has_communication:
                    logger.debug('There is no communication profile in this NPU run.')
                    return None
            generator = DistributedRunGenerator(profiles, span, self.device_target)
            profile = generator.generate_run_profile()
            return profile

    def _gpu_distributed(self, profiles, span):
        has_communication = True
        comm_node_lists: List[List[CommunicationNode]] = []
        for data in profiles:
            logger.debug('Processing profile data')
            # Set has_communication to False and disable distributed view if any one worker has no communication
            if data.has_communication and data.comm_node_list:
                comm_node_lists.append(data.comm_node_list)
                if len(comm_node_lists[-1]) != len(comm_node_lists[0]):
                    logger.error("Number of communication operation nodes don't match between workers in run: %s"
                                 % self.run_name)
                    has_communication = False
            else:
                has_communication = False
            logger.debug('Processing profile data finish')

        if not has_communication:
            logger.debug('There is no communication profile in this GPU run.')
            return None

        worker_num = len(comm_node_lists)
        for i, node in enumerate(comm_node_lists[0]):
            kernel_range_size = len(node.kernel_ranges)
            # loop for all communication kernel ranges in order
            for j in range(kernel_range_size):
                min_range = sys.maxsize
                # For each kernel_range, find the minist between workers as the real communication time
                for k in range(worker_num):
                    kernel_ranges = comm_node_lists[k][i].kernel_ranges
                    if len(kernel_ranges) != kernel_range_size:
                        logger.error("Number of communication kernels don't match between workers in run: %s"
                                     % self.run_name)
                        has_communication = False
                        return None
                    if kernel_ranges:
                        if kernel_ranges[j][1] - kernel_ranges[j][0] < min_range:
                            min_range = kernel_ranges[j][1] - kernel_ranges[j][0]
                for k in range(worker_num):
                    kernel_range = comm_node_lists[k][i].kernel_ranges[j]
                    comm_node_lists[k][i].real_time_ranges.append((kernel_range[1] - min_range, kernel_range[1]))

        for data in profiles:
            data.communication_parse()

        generator = DistributedRunGenerator(profiles, span, self.device_target)
        profile = generator.generate_run_profile()
        return profile
