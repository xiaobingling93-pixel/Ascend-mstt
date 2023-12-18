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
import csv
import json
import re
import io as sysio
from collections import OrderedDict, defaultdict
from json import JSONDecodeError
from typing import Dict, Iterable, List
import numpy as np

from .. import consts, utils, io
from ..run import DistributedRunProfile, RunProfile
from .data import DistributedRunProfileData, RunProfileData
from .module_op import aggegate_module_view, aggegate_pl_module_view
from .op_agg import KernelAggByNameOp, OperatorAgg
from .overall_parser import ProfileRole
from ..utils import Canonicalizer

logger = utils.get_logger()


class RunGenerator(object):
    def __init__(self, worker, span, profile_data: RunProfileData, device_target="GPU"):
        self.worker = worker
        self.span = span
        self.profile_data = profile_data
        self.statistic_data = {}
        self.accelerator_data = {}
        self.device_target = device_target
        self.component_curve_data = {}
        self.process_data = {}

    def generate_run_profile(self):
        profile_run = RunProfile(self.worker, self.span)
        profile_run.is_pytorch_lightning = self.profile_data.is_pytorch_lightning
        profile_run.has_runtime = self.profile_data.has_runtime
        profile_run.has_kernel = self.profile_data.has_kernel
        profile_run.has_communication = self.profile_data.has_communication
        profile_run.has_memcpy_or_memset = self.profile_data.has_memcpy_or_memset
        profile_run.profiler_start_ts = self.profile_data.profiler_start_ts
        profile_run.device_target = self.device_target

        if self.device_target != 'Ascend':
            profile_run.views.append(consts.OVERALL_VIEW)
            profile_run.overview = self._generate_overview()

            profile_run.views.append(consts.OP_VIEW)
            profile_run.operation_pie_by_name = self._generate_op_pie()
            profile_run.operation_table_by_name = self._generate_op_table(self.profile_data.op_list_groupby_name)
            profile_run.operation_stack_by_name = self._generate_op_table_for_stack(False)
            profile_run.operation_pie_by_name_input = self._generate_op_pie(True)
            profile_run.operation_table_by_name_input = self._generate_op_table(
                self.profile_data.op_list_groupby_name_input, True)
            profile_run.operation_stack_by_name_input = self._generate_op_table_for_stack(True)

            if self.profile_data.has_kernel:
                profile_run.views.append(consts.KERNEL_VIEW)
                profile_run.kernel_table = self._generate_kernel_table_gpu()
                profile_run.kernel_op_table = self._generate_kernel_op_table_gpu()
                profile_run.kernel_pie = self._generate_kernel_pie_gpu()
                profile_run.tc_pie = self._generate_tc_pie_gpu()

            if self.profile_data.memory_snapshot:
                profile_run.views.append(consts.MEMORY_VIEW)
                profile_run.memory_snapshot = self.profile_data.memory_snapshot

            profile_run.gpu_metrics = self.profile_data.gpu_metrics_parser.get_gpu_metrics()

            gpu_infos = {gpu_id: RunGenerator._get_gpu_info(self.profile_data.device_props, gpu_id)
                         for gpu_id in self.profile_data.gpu_metrics_parser.gpu_ids}
            gpu_infos = {gpu_id: gpu_info for gpu_id, gpu_info in gpu_infos.items() if gpu_info is not None}

            profile_run.gpu_summary, profile_run.gpu_tooltip = \
                self.profile_data.gpu_metrics_parser.get_gpu_metrics_data_tooltip(
                    gpu_infos, self.profile_data.tc_ratio)

            profile_run.pl_tid2tree = self.profile_data.pl_tid2tree

            profile_run.module_stats = aggegate_module_view(self.profile_data.tid2tree, self.profile_data.events)
            profile_run.pl_module_stats = aggegate_pl_module_view(self.profile_data.tid2tree, self.profile_data.events)
            if profile_run.is_pytorch_lightning and profile_run.pl_module_stats:
                profile_run.views.append(consts.LIGHTNING_VIEW)
            elif profile_run.module_stats:
                profile_run.views.append(consts.MODULE_VIEW)
        else:
            if self.profile_data.has_operator_view:
                profile_run.views.append(consts.OP_VIEW)
                profile_run.operation_pie_by_name = self._get_operator_pie()
                profile_run.operation_table_by_name = self._get_operator_table_by_name()
                profile_run.operation_stack_by_name = self._get_call_stack_by_name()
                profile_run.operation_pie_by_name_input = self._get_operator_pie(True)
                profile_run.operation_table_by_name_input = self._get_operator_table_by_name(True)
                profile_run.operation_stack_by_name_input = self._get_call_stack_by_name_shapes(True)

            if self.profile_data.has_kernel:
                profile_run.views.append(consts.KERNEL_VIEW)
                profile_run.kernel_table = self._generate_kernel_table_npu()
                profile_run.kernel_op_table = self._generate_kernel_op_table_npu()
                profile_run.kernel_pie = self._generate_kernel_pie_npu()
                profile_run.tc_pie = self._generate_tc_pie_npu()

            if self.profile_data.has_memory:
                profile_run.views.append(consts.MEMORY_VIEW)
                profile_run.memory_div_curve = None
                self.process_data, self.component_curve_data = self._handle_memory_data()
                profile_run.memory_all_curve = self._get_memory_all_curve()
                peak_memory_events = self._handle_memory_component()
                profile_run.memory_events = self._get_memory_event(peak_memory_events)

            if self.profile_data.has_communication:
                profile_run.step_to_overlap = self._npu_get_overlap()
                profile_run.step_to_wait, profile_run.comm_op = self._npu_get_wait_table()

        profile_run.tid2tree = self.profile_data.tid2tree
        if self.profile_data.has_trace:
            profile_run.views.append(consts.TRACE_VIEW)
            profile_run.trace_file_path = self.profile_data.trace_file_path

        return profile_run

    def _npu_get_overlap(self):
        path = self.profile_data.distributed_csv_path
        overlap_by_steps: Dict[str, List[float]] = OrderedDict()
        data = RunGenerator._get_csv_data(path)
        if len(data) <= 1:
            return overlap_by_steps
        title = [x.lower() for x in data[0]]
        title_name = RunGenerator._check_overlap_data(title)
        if not title_name:
            logger.error("Incomplete content of CSV file.")
            return overlap_by_steps

        for step in data[1:]:
            key = step[0]
            if key == '':
                key = 'all'
            overlap = [float(step[int(title_name[0])]), float(step[int(title_name[1])]),
                       float(step[int(title_name[2])]), float(step[int(title_name[3])])]
            if key in overlap_by_steps:
                overlap_by_steps[key] = list(np.add(overlap, overlap_by_steps[key]))
            else:
                overlap_by_steps[key] = list(overlap)
        return overlap_by_steps

    @staticmethod
    def _check_overlap_data(title):
        # csv: step / compute time / communication_not_overlap / overlap / communication / free time
        length = len(title)
        if length < 5:
            return []
        key = ["computing", "overlapped", "communication(not overlapped)", "free"]
        get_key = list()
        for j in key:
            for i in range(length):
                if j == title[i]:
                    get_key.append(i)
        if len(get_key) < 4:
            return []
        return get_key

    def _npu_get_wait_table(self):
        path = self.profile_data.communication_json_path
        if not io.exists(path):
            raise FileNotFoundError(path)
        data = io.read(path)
        try:
            communication_json = json.loads(data, strict=False)
        except JSONDecodeError as e:
            try:
                communication_json = json.loads(data, strict=False)
            except JSONDecodeError:
                with sysio.StringIO() as fout:
                    str_data = data.decode('utf-8')
                    # only replace the N/A without surrounding double quote
                    fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data))
                    communication_json = json.loads(fout.getvalue())
                    logger.warning('Get JSONDecodeError: %s, Re-encode it to temp file' % e.msg)

        wait_by_step: Dict[str, Dict[str, float]] = OrderedDict()
        table_ops: Dict[str, List[float]] = OrderedDict()
        if len(communication_json) <= 0:
            return wait_by_step, table_ops
        for step in communication_json:
            step_id = re.sub(r'step', '', step)
            if step_id == '':
                step_id = 'all'
            data = communication_json.get(step)
            collection_ops = data.get("collective")
            p2p_ops = data.get("p2p")
            coll_total_trans, coll_total_synchronize = RunGenerator._get_wait_table_by_ops(collection_ops, table_ops)
            p2p_total_trans, p2p_total_synchronize = RunGenerator._get_wait_table_by_ops(p2p_ops, table_ops)

            wait_by_step[step_id] = {
                "trans": coll_total_trans + p2p_total_trans,
                "Synchronize": coll_total_synchronize + p2p_total_synchronize
            }
        return wait_by_step, table_ops

    @staticmethod
    def _get_wait_table_by_ops(op, ops):
        total_trans = 0
        total_synchronize = 0
        for key, data in op.items():
            if str(key) == "Total Op Info" and data.get("Communication Time Info"):
                total_trans += float(data.get("Communication Time Info").get("Transit Time(ms)"))
                total_synchronize += float(data.get("Communication Time Info").get("Synchronization Time(ms)"))
                continue
            k = re.sub(r'[0-9]+', ' ', key).split(" ")[0]
            if k not in ops:
                ops[k] = [0, 0, 0, 0]
            ops[k][0] += 1
            for _, band in data.get("Communication Bandwidth Info").items():
                ops[k][1] += float(band.get("Transit Size(MB)"))
            if data.get("Communication Time Info") is not None:
                ops[k][2] += data.get("Communication Time Info").get("Elapse Time(ms)")
                ops[k][3] += data.get("Communication Time Info").get("Transit Time(ms)")
        return total_trans, total_synchronize

    def _get_operator_details_by_name(self):
        operator_by_name = defaultdict(list)
        operator_by_name_and_input_shapes = defaultdict(list)
        path = self.profile_data.operator_path
        datas = RunGenerator._get_csv_data(path)
        if len(datas) <= 1:
            return operator_by_name, operator_by_name_and_input_shapes
        for ls in datas[1:]:
            try:
                temp: list = [ls[0], RunGenerator._trans_shape(str(ls[1])), ls[2], float(ls[3]), float(ls[4]),
                              float(ls[5]), float(ls[6]), float(ls[7]), float(ls[8])]
            except (ValueError, IndexError):
                logger.error('Data in file "operator_details.csv" has wrong format.')
                return operator_by_name, operator_by_name_and_input_shapes
            operator_by_name[ls[0]].append(temp)
            key = "{}###{}".format(str(ls[0]), RunGenerator._trans_shape(str(ls[1])))
            operator_by_name_and_input_shapes[key].append(temp)
        return operator_by_name, operator_by_name_and_input_shapes

    def _get_operator_table_by_name(self, group_by_input_shape=False):
        result = {
            'metadata': {
                'sort': 'device_self_duration',
                'tooltips': {
                    'tc_eligible': consts.TOOLTIP_OP_TC_ELIGIBLE_AICORE,
                    'tc_self_ratio': consts.TOOLTIP_OP_TC_SELF_AICORE,
                    'tc_total_ratio': consts.TOOLTIP_OP_TC_TOTAL_AICORE
                }
            },
            'data': self._set_operator_data(True) if group_by_input_shape else self._set_operator_data()
        }
        return result

    def _get_operator_pie(self, group_by_input_shape=False):
        data = {}
        tag = {'device_self_time': 'Device Self Time (us)', 'device_total_time': 'Device Total Time (us)',
               'host_self_time': 'Host Self Time (us)', 'host_total_time': 'Host Total Time (us)'}
        for key, value in tag.items():
            data[key] = {
                'title': value,
                'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}],
                'rows': []
            }
        for value in iter(self._set_operator_data(group_by_input_shape)
                          if group_by_input_shape else self._set_operator_data()):
            data['device_self_time'].get('rows').append([value.get('name'), value.get('device_self_duration')])
            data['device_total_time'].get('rows').append([value.get('name'), value.get('device_total_duration')])
            data['host_self_time'].get('rows').append([value.get('name'), value.get('host_self_duration')])
            data['host_total_time'].get('rows').append([value.get('name'), value.get('host_total_duration')])
        return data

    def _set_operator_data(self, group_by_input_shape=False):
        result = []
        if group_by_input_shape:
            _, operator_by_name = self._get_operator_details_by_name()
        else:
            operator_by_name, _ = self._get_operator_details_by_name()
        for name_key, values in operator_by_name.items():
            if group_by_input_shape:
                name = name_key.split("###")[0]
                shape = name_key.split("###")[1]
                result.append(RunGenerator._get_table_head(name, shape, None, values))
            else:
                result.append(RunGenerator._get_table_head(name_key, None, None, values))
        return result

    def _set_name_callstack_data(self, group_by_input_shape=False):
        if group_by_input_shape:
            _, operator_by_name = self._get_operator_details_by_name()
        else:
            operator_by_name, _ = self._get_operator_details_by_name()

        result = dict()
        for key, values in operator_by_name.items():
            name_callstack = defaultdict(list)
            for value in iter(values):
                name_callstack[str(value[2])].append(value)
            result[key] = name_callstack
        return result

    def _get_call_stack_by_name_shapes(self, group_by_input_shape: bool = False):
        result = dict()
        name_input_shapes_callstack_data = self._set_name_callstack_data(group_by_input_shape)
        for name_key, values in name_input_shapes_callstack_data.items():
            name = name_key.split("###")[0]
            shape = name_key.split("###")[1]
            table = {
                'metadata': {
                    'sort': 'device_self_duration',
                    'tooltips': {
                        'tc_eligible': consts.TOOLTIP_OP_TC_ELIGIBLE_AICORE,
                        'tc_self_ratio': consts.TOOLTIP_OP_TC_SELF_AICORE,
                        'tc_total_ratio': consts.TOOLTIP_OP_TC_TOTAL_AICORE
                    }
                },
                'data': []
            }
            for callstack_key, value in values.items():
                table['data'].append(RunGenerator._get_table_head(name, shape, callstack_key, value))
            result[name_key] = table
        return result

    @staticmethod
    def _trans_shape(shape: str):
        result = list()
        if ';' not in shape:
            result.append('[' + shape.strip() + ']')
            return '[' + ', '.join(result) + ']'
        if len(shape.strip()) <= 1:
            result.append('[]')
            return '[' + ', '.join(result) + ']'
        shape_spl = shape.split("\n")
        for shape_div in iter(shape_spl):
            result.append('[' + str(shape_div.replace(';', '')) + ']')
        return '[' + ', '.join(result) + ']'

    def _get_call_stack_by_name(self):
        result = dict()
        name_callstack_data = self._set_name_callstack_data()
        for name_key, values in name_callstack_data.items():
            table = {
                'metadata': {
                    'sort': 'device_self_duration',
                    'tooltips': {
                        'tc_eligible': consts.TOOLTIP_OP_TC_ELIGIBLE_AICORE,
                        'tc_self_ratio': consts.TOOLTIP_OP_TC_SELF_AICORE,
                        'tc_total_ratio': consts.TOOLTIP_OP_TC_TOTAL_AICORE
                    }
                },
                'data': []
            }
            for callstack_key, value in values.items():
                table['data'].append(RunGenerator._get_table_head(name_key, None, callstack_key, value))
            result[name_key] = table
        return result

    @staticmethod
    def _get_table_head(name: str, input_shape: str, call_stack: str, value: list):
        if name is None:
            return {}
        temp = {'name': name, 'calls': 0, 'host_self_duration': 0,
                'host_total_duration': 0, 'device_self_duration': 0, 'device_total_duration': 0,
                'tc_self_ratio': 0, 'tc_total_ratio': 0, 'tc_eligible': 'Yes'}
        if input_shape is not None:
            temp['input_shape'] = input_shape
            if call_stack is not None:
                temp['call_stack'] = call_stack
            else:
                temp['has_call_stack'] = False
        else:
            if call_stack is not None:
                temp['call_stack'] = call_stack
            else:
                temp['has_call_stack'] = False
        for vl in iter(value):
            if 'has_call_stack' in temp and vl[2]:
                temp['has_call_stack'] = True
            temp['calls'] += 1
            temp['host_self_duration'] = round(temp['host_self_duration'] + vl[3], 2)
            temp['host_total_duration'] = round(temp['host_total_duration'] + vl[4], 2)
            temp['device_self_duration'] = round(temp['device_self_duration'] + vl[5], 2)
            temp['device_total_duration'] = round(temp['device_total_duration'] + vl[6], 2)
            temp['tc_self_ratio'] = round(temp['tc_self_ratio'] + vl[7], 2)
            temp['tc_total_ratio'] = round(temp['tc_total_ratio'] + vl[8], 2)
        temp['tc_eligible'] = 'Yes' if temp['tc_self_ratio'] > 0 or temp['tc_total_ratio'] > 0 else 'No'
        temp['tc_self_ratio'] = 0 if temp['device_self_duration'] == 0 \
            else round(temp['tc_self_ratio'] / temp['device_self_duration'] * 100, 2)
        temp['tc_total_ratio'] = 0 if temp['device_total_duration'] == 0 \
            else round(temp['tc_total_ratio'] / temp['device_total_duration'] * 100, 2)
        return temp

    def _get_memory_event(self, peak_memory_events: dict):
        display_columns = ('Name', 'Size(KB)', 'Allocation Time(us)', 'Release Time(us)', 'Duration(us)')
        path = self.profile_data.memory_operator_path
        display_datas = defaultdict(list)
        devices_type = []
        table = {
            'metadata': {
                'title': 'Memory Events',
                'default_device': 'all',
            },
            'columns': [],
            'rows': {}
        }
        datas = RunGenerator._get_csv_data(path)
        for idx, column in enumerate(datas[0]):
            if column == 'Device Type':
                self.device_type_form_idx = idx
            if column in display_columns:
                if column == 'Name':
                    table['columns'].append({'name': column, 'type': 'string'})
                elif column == 'Size(KB)':
                    table['columns'].append({'name': column, 'type': 'number'})
                else:
                    # Convert time metric
                    table['columns'].append({'name': column.replace('(us)', '(ms)'), 'type': 'number'})
        for ls in datas[1:]:
            device_type = ls[self.device_type_form_idx]
            # convert time metric 'us' to 'ms'
            # some operators may not have the following columns
            nums = [ls[0] if ls[0] else '<unknown>', abs(float(ls[1])),
                    round((float(ls[2]) - self.profile_data.profiler_start_ts) / 1000, 3) if ls[2] else None,
                    round((float(ls[3]) - self.profile_data.profiler_start_ts) / 1000, 3) if ls[3] else None,
                    round(float(ls[4]) / 1000, 3) if ls[4] else None]
            display_datas[device_type].append(nums)
        table['rows'] = display_datas
        for name in display_datas:
            devices_type.append(name)
        table['metadata'].update({'default_device': devices_type[0]})
        return {
            'operator': table,
            'component': peak_memory_events
        }

    def _get_memory_all_curve(self):
        time_metric: str = 'ms'
        memory_metric: str = 'MB'
        cano = Canonicalizer(time_metric, memory_metric)
        process_devices_type, process_peaks = RunGenerator._get_process_peaks_and_devices_type(self.process_data,
                                                                                               memory_metric)
        total_result = {
            'metadata': {
                'devices': process_devices_type,
                'default_device': process_devices_type[0] if len(process_devices_type) > 0 else '',
                'peaks': process_peaks,
                'totals': {},
                'first_ts': 0,
                'time_metric': cano.time_metric,
                'memory_metric': cano.memory_metric,
                'time_factor': cano.time_factor,
                'memory_factor': cano.memory_factor,
            },
            'columns': defaultdict(list),
            'rows': self.process_data
        }
        for device in process_devices_type:
            if self.process_data.get(device).get('Allocated') is not None and self.process_data.get(device).get(
                    'Reserved') is not None:
                total_result['columns'][device].append(
                    {'name': f'Allocated ({cano.memory_metric})', 'type': 'number', 'tooltip': 'PTA+GE memory in use.'})
                total_result['columns'][device].append(
                    {'name': f'Reserved ({cano.memory_metric})', 'type': 'number',
                     'tooltip': 'PTA+GE reserved memory by allocator, both used and unused.'})
            if len(total_result['columns'][device]) > 0:
                total_result['columns'][device].insert(0, {'name': f'Time ({cano.time_metric})', 'type': 'number',
                                                           'tooltip': 'Time since profiler starts.'})
        pta_ge_devices_type, pta_ge_peaks = RunGenerator._get_pta_ge_peaks_and_devices_type(self.component_curve_data,
                                                                                            memory_metric)
        component_curve_result = {
            'metadata': {
                'devices': pta_ge_devices_type,
                'default_device': pta_ge_devices_type[0] if len(pta_ge_devices_type) > 0 else '',
                'peaks': pta_ge_peaks,
                'totals': {},
                'first_ts': 0,
                'time_metric': cano.time_metric,
                'memory_metric': cano.memory_metric,
                'time_factor': cano.time_factor,
                'memory_factor': cano.memory_factor,
            },
            'columns': defaultdict(list),
            'rows': self.component_curve_data
        }
        for device in pta_ge_devices_type:
            if self.component_curve_data.get(device).get('PTA') is not None:
                component_curve_result['columns'][device] += [
                    {'name': f'PTA Allocated ({cano.memory_metric})', 'type': 'number',
                     'tooltip': 'PTA memory in use.'},
                    {'name': f'PTA Reserved ({cano.memory_metric})', 'type': 'number',
                     'tooltip': 'PTA reserved memory by allocator, both used and unused.'}]
            if self.component_curve_data.get(device).get('GE') is not None:
                component_curve_result['columns'][device] += [
                    {'name': f'GE Allocated ({cano.memory_metric})', 'type': 'number', 'tooltip': 'GE memory in use.'},
                    {'name': f'GE Reserved ({cano.memory_metric})', 'type': 'number',
                     'tooltip': 'GE reserved memory by allocator, both used and unused.'}]
            if self.component_curve_data.get(device).get('APP') is not None:
                component_curve_result['columns'][device] += [
                    {'name': f'APP Reserved ({cano.memory_metric})', 'type': 'number',
                     'tooltip': 'APP reserved memory by allocator, both used and unused.'}]
            if len(component_curve_result['columns'][device]) > 0:
                component_curve_result['columns'][device].insert(0, {'name': f'Time ({cano.time_metric})',
                                                                     'type': 'number',
                                                                     'tooltip': 'Time since profiler starts.'})
        device_types = list(set(process_devices_type + pta_ge_devices_type))
        return {
            'devices': device_types,
            'default_device': device_types[0],
            'total': total_result,
            'ptaGe': component_curve_result
        }

    @staticmethod
    def _get_process_peaks_and_devices_type(process_data: dict, memory_metric: str):
        devices_type = []
        peaks = {}
        for device in process_data:
            devices_type.append(device)
            reserved_list = process_data.get(device).get('Allocated')
            if reserved_list is not None:
                max_reserved = 0
                for array_value in reserved_list:
                    max_reserved = max(array_value[1], max_reserved)
                peaks[device] = f'Peak Memory Usage: {max_reserved:.1f}{memory_metric}'
        return devices_type, peaks

    @staticmethod
    def _get_pta_ge_peaks_and_devices_type(process_data: dict, memory_metric: str):
        devices_type = []
        peaks = {}
        for device in process_data:
            devices_type.append(device)
            peaks[device] = 'Reserved Peak Memory Usage:'
            for component in process_data.get(device):
                max_reserved = 0
                for array_value in process_data.get(device).get(component):
                    max_reserved = max(array_value[2], max_reserved)
                peaks[device] += f' {component}-{max_reserved:.1f}{memory_metric} |'
        return devices_type, peaks

    @staticmethod
    def _check_csv_columns(columns: list, column_idxs: dict):
        column_exist_count = 0
        for idx, column in enumerate(columns):
            if column in column_idxs:
                column_idxs[column] = idx
                column_exist_count += 1
        return column_idxs.values(), column_exist_count

    def _handle_memory_data(self):
        process_data = defaultdict()
        pta_or_ge_data = defaultdict()
        datas = RunGenerator._get_csv_data(self.profile_data.memory_curve_path)
        required_column_idxs = {
            'Component': -1,
            'Device Type': -1,
            'Timestamp(us)': -1,
            'Total Reserved(MB)': -1,
            'Total Allocated(MB)': -1
        }
        (tag_type_idx, device_type_idx, time_idx, reserved_idx, allocated_idx), column_exist_count = \
            RunGenerator._check_csv_columns(datas[0], required_column_idxs)
        if column_exist_count < len(required_column_idxs):
            logger.error('Required column is missing in file "memory_record.csv"')
        else:
            for ls in datas[1:]:
                time_column = round((float(ls[time_idx]) - self.profile_data.profiler_start_ts) / 1000, 3)
                device_type = ls[device_type_idx]
                if ls[tag_type_idx] == 'PTA+GE':
                    process_data.setdefault(device_type, {}).setdefault('Allocated', []).append(
                        [time_column, round(float(ls[allocated_idx]), 3)])
                    process_data.setdefault(device_type, {}).setdefault('Reserved', []).append(
                        [time_column, round(float(ls[reserved_idx]), 3)])
                elif ls[tag_type_idx] == 'APP':
                    line_chart_data = [time_column, None, round(float(ls[reserved_idx]), 3)]
                    pta_or_ge_data.setdefault(device_type, {}).setdefault(ls[tag_type_idx], []).append(line_chart_data)
                elif ls[tag_type_idx] in ('PTA', 'GE'):
                    line_chart_data = [time_column, round(float(ls[allocated_idx]), 3),
                                       round(float(ls[reserved_idx]), 3)]
                    pta_or_ge_data.setdefault(device_type, {}).setdefault(ls[tag_type_idx], []).append(line_chart_data)

        return process_data, pta_or_ge_data

    def _handle_memory_component(self):
        peak_memory_events = {
            'metadata': {
                'title': 'Component Peak Memory',
                'default_device': '',
            },
            'columns': [{'name': 'Component', 'type': 'string'},
                        {'name': 'Peak Memory Reserved(MB)', 'type': 'number'},
                        {'name': 'Time(ms)', 'type': 'number'}]
        }
        peak_memory_rows = defaultdict(list)
        component_datas = RunGenerator._get_csv_data(self.profile_data.memory_component_path)
        if component_datas:
            required_column_idxs = {
                'Component': -1,
                'Timestamp(us)': -1,
                'Total Reserved(MB)': -1,
                'Device': -1
            }
            (tag_type_idx, time_idx, reserved_idx, device_type_idx), column_exist_count = \
                RunGenerator._check_csv_columns(component_datas[0], required_column_idxs)
            if column_exist_count < len(required_column_idxs):
                logger.error('Required column is missing in file "npm_module_mem.csv"')
            else:
                for ls in component_datas[1:]:
                    memory_curve_id_dict = {
                        'device_type_idx': device_type_idx,
                        'reserved_idx': reserved_idx,
                        'tag_type_idx': tag_type_idx,
                        'time_idx': time_idx
                    }
                    self._handle_peak_memory_rows(memory_curve_id_dict, ls, peak_memory_rows)
        peak_memory_events['rows'] = peak_memory_rows
        return peak_memory_events

    def _handle_peak_memory_rows(self, memory_curve_id_dict, ls, peak_memory_rows):
        # Record the peak memory usage of other components.
        has_flag = False
        device_type_idx = memory_curve_id_dict.get('device_type_idx')
        reserved_idx = memory_curve_id_dict.get('reserved_idx')
        tag_type_idx = memory_curve_id_dict.get('tag_type_idx')
        time_idx = memory_curve_id_dict.get('time_idx')
        time_column = round((float(ls[time_idx]) - self.profile_data.profiler_start_ts) / 1000, 3)
        for item in peak_memory_rows[ls[device_type_idx]]:
            if item[0] == ls[tag_type_idx]:
                if item[1] < ls[reserved_idx]:
                    item[1] = ls[reserved_idx]
                    item[2] = time_column
                elif item[1] == ls[reserved_idx]:
                    item[2] = min(item[2], time_column)
                has_flag = True
                break
        if not has_flag:
            peak_memory_rows[ls[device_type_idx]].append([ls[tag_type_idx], ls[reserved_idx], time_column])

    def _generate_overview(self):
        def build_part_time_str(part_cost: float, part_name: str):
            format_str = ('<div class="visualization-tooltip" style="white-space: nowrap;">'
                          'Step {}<br>'
                          'Total: {}us<br>'
                          '<b>{}: {}us</b><br>'
                          'Percentage: {}%'
                          '</div>')
            percentage = round(100 * part_cost / costs.costs[ProfileRole.Total], 2)
            return format_str.format(step_name, costs.costs[ProfileRole.Total], part_name, part_cost, percentage)

        def build_avg_cost_dict(part_name: str, part_cost: float):
            cost_dict = {'name': part_name,
                         'description': '',
                         'value': round(part_cost),
                         'extra': round(100 * part_cost / self.profile_data.avg_costs.costs[ProfileRole.Total], 2)}
            return cost_dict

        show_gpu = (self.profile_data.has_runtime
                    or self.profile_data.has_kernel or self.profile_data.has_memcpy_or_memset)

        column_tootip = {'type': 'string', 'role': 'tooltip', 'p': {'html': 'true'}}
        data = {}
        data['steps'] = {}
        data['steps']['columns'] = ['Step']
        if show_gpu:
            data['steps']['columns'].extend(['Kernel', 'Memcpy', 'Memset'])
        if self.profile_data.has_communication:
            data['steps']['columns'].append('Communication')
        if show_gpu:
            data['steps']['columns'].append('Runtime')
        data['steps']['columns'].extend(['DataLoader', 'CPU Exec', 'Other'])

        data['steps']['rows'] = []
        for i in range(len(self.profile_data.steps_costs)):
            costs = self.profile_data.steps_costs[i]
            step_name = self.profile_data.steps_names[i]
            row = [{'value': step_name}]
            if show_gpu:
                row.extend([{'value': costs.costs[ProfileRole.Kernel],
                             'tooltip': build_part_time_str(costs.costs[ProfileRole.Kernel], 'Kernel')},
                            {'value': costs.costs[ProfileRole.Memcpy],
                             'tooltip': build_part_time_str(costs.costs[ProfileRole.Memcpy], 'Memcpy')},
                            {'value': costs.costs[ProfileRole.Memset],
                             'tooltip': build_part_time_str(costs.costs[ProfileRole.Memset], 'Memset')}])
            if self.profile_data.has_communication:
                row.append({'value': costs.costs[ProfileRole.Communication],
                            'tooltip': build_part_time_str(costs.costs[ProfileRole.Communication], 'Communication')})
            if show_gpu:
                row.append({'value': costs.costs[ProfileRole.Runtime],
                            'tooltip': build_part_time_str(costs.costs[ProfileRole.Runtime], 'Runtime')})
            row.extend([{'value': costs.costs[ProfileRole.DataLoader],
                         'tooltip': build_part_time_str(costs.costs[ProfileRole.DataLoader], 'DataLoader')},
                        {'value': costs.costs[ProfileRole.CpuOp],
                         'tooltip': build_part_time_str(costs.costs[ProfileRole.CpuOp], 'CPU Exec')},
                        {'value': costs.costs[ProfileRole.Other],
                         'tooltip': build_part_time_str(costs.costs[ProfileRole.Other], 'Other')}])
            data['steps']['rows'].append(row)

        avg_costs = []
        if show_gpu:
            avg_costs.extend([
                build_avg_cost_dict('Kernel', self.profile_data.avg_costs.costs[ProfileRole.Kernel]),
                build_avg_cost_dict('Memcpy', self.profile_data.avg_costs.costs[ProfileRole.Memcpy]),
                build_avg_cost_dict('Memset', self.profile_data.avg_costs.costs[ProfileRole.Memset])
            ])
        if self.profile_data.has_communication:
            avg_costs.extend([
                build_avg_cost_dict('Communication', self.profile_data.avg_costs.costs[ProfileRole.Communication])
            ])
        if show_gpu:
            avg_costs.extend([
                build_avg_cost_dict('Runtime', self.profile_data.avg_costs.costs[ProfileRole.Runtime])
            ])
        avg_costs.extend([
            build_avg_cost_dict('DataLoader', self.profile_data.avg_costs.costs[ProfileRole.DataLoader]),
            build_avg_cost_dict('CPU Exec', self.profile_data.avg_costs.costs[ProfileRole.CpuOp]),
            build_avg_cost_dict('Other', self.profile_data.avg_costs.costs[ProfileRole.Other])
        ])

        data['performance'] = [{'name': 'Average Step Time', 'description': '',
                                'value': round(self.profile_data.avg_costs.costs[ProfileRole.Total]),
                                'extra': 100, 'children': avg_costs}]

        if len(self.profile_data.recommendations) == 0:
            html = '<li>N/A</li>'
        else:
            html = ''
            for recommendation in self.profile_data.recommendations:
                html += '<li>{}</li>'.format(recommendation)
        data['recommendations'] = '<ul>{}</ul>'.format(html)

        return data

    def _generate_op_pie(self, group_by_input_shape: bool = False):
        op_device_total_time = []
        op_device_self_time = []
        op_host_total_time = []
        op_host_self_time = []

        if group_by_input_shape:
            op_list = self.profile_data.op_list_groupby_name_input
        else:
            op_list = self.profile_data.op_list_groupby_name

        for op_agg in op_list:
            # Whether device_duration & self_device_duration are accurate or not depends on the input tracing data.
            if op_agg.device_duration > 0:
                op_device_total_time.append([op_agg.name, op_agg.device_duration])
            if op_agg.self_device_duration > 0:
                op_device_self_time.append([op_agg.name, op_agg.self_device_duration])
            if op_agg.host_duration > 0:
                op_host_total_time.append([op_agg.name, op_agg.host_duration])
            if op_agg.self_host_duration > 0:
                op_host_self_time.append([op_agg.name, op_agg.self_host_duration])

        op_device_total_time.sort(key=lambda x: x[1], reverse=True)
        op_device_self_time.sort(key=lambda x: x[1], reverse=True)
        op_host_total_time.sort(key=lambda x: x[1], reverse=True)
        op_host_self_time.sort(key=lambda x: x[1], reverse=True)

        data = {}
        device_total_time = {}
        device_self_time = {}
        host_total_time = {}
        host_self_time = {}

        if len(op_device_total_time) > 0:
            device_total_time['title'] = 'Device Total Time (us)'
            device_total_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            device_total_time['rows'] = op_device_total_time
        else:
            device_total_time = None

        if len(op_device_self_time) > 0:
            device_self_time['title'] = 'Device Self Time (us)'
            device_self_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            device_self_time['rows'] = op_device_self_time
        else:
            device_self_time = None

        if len(op_host_total_time) > 0:
            host_total_time['title'] = 'Host Total Time (us)'
            host_total_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            host_total_time['rows'] = op_host_total_time
        else:
            host_total_time = None

        if len(op_host_self_time) > 0:
            host_self_time['title'] = 'Host Self Time (us)'
            host_self_time['columns'] = [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}]
            host_self_time['rows'] = op_host_self_time
        else:
            host_self_time = None

        data['device_total_time'] = device_total_time
        data['device_self_time'] = device_self_time
        data['host_total_time'] = host_total_time
        data['host_self_time'] = host_self_time

        return data

    def _generate_op_table(self, op_list: Iterable[OperatorAgg], group_by_input_shape=False, call_stack=False):
        show_gpu = self.profile_data.has_kernel or self.profile_data.has_memcpy_or_memset

        if group_by_input_shape:
            stack_list_dict = self.profile_data.stack_lists_group_by_name_input
        else:
            stack_list_dict = self.profile_data.stack_lists_group_by_name

        op_list = sorted(op_list,
                         key=lambda x: x.self_device_duration if show_gpu else x.self_host_duration,
                         reverse=True)

        data = list()
        result = {
            'metadata': {
                'sort': 'device_self_duration' if show_gpu else 'host_self_duration',
                'tooltips': {
                    'tc_eligible': consts.TOOLTIP_OP_TC_ELIGIBLE,
                    'tc_self_ratio': consts.TOOLTIP_OP_TC_SELF,
                    'tc_total_ratio': consts.TOOLTIP_OP_TC_TOTAL
                }
            },
            'data': data
        }
        for op in op_list:
            # Whether device_duration & self_device_duration are accurate or not depends on the input tracing data.
            row = dict()
            row['name'] = op.name
            if group_by_input_shape:
                row['input_shape'] = op.input_shape
            row['calls'] = op.calls
            if show_gpu:
                row['device_self_duration'] = round(op.self_device_duration)
                row['device_total_duration'] = round(op.device_duration)
            row['host_self_duration'] = round(op.self_host_duration)
            row['host_total_duration'] = round(op.host_duration)
            row['tc_eligible'] = 'Yes' if op.tc_eligible else 'No'
            row['tc_self_ratio'] = round(100 * op.tc_self_ratio, 2)
            row['tc_total_ratio'] = round(100 * op.tc_total_ratio, 2)
            if call_stack:
                row['call_stack'] = op.callstacks.pop()
            else:
                if group_by_input_shape:
                    key = op.name + '###' + str(op.input_shape)
                else:
                    key = op.name
                row['has_call_stack'] = key in stack_list_dict
            data.append(row)

        return result

    def _generate_op_table_for_stack(self, group_by_input_shape: bool):
        if group_by_input_shape:
            stack_list_dict = self.profile_data.stack_lists_group_by_name_input
        else:
            stack_list_dict = self.profile_data.stack_lists_group_by_name

        result = dict()
        for k, v in stack_list_dict.items():
            result[k] = self._generate_op_table(v, group_by_input_shape, True)
        return result

    def _generate_kernel_op_table_gpu(self):
        table = {}
        result = {
            'metadata': {
                'sort': 'Total Duration (us)'
            },
            'data': table
        }
        table['columns'] = [{'type': 'string', 'name': 'Name'},
                            {'type': 'string', 'name': 'Operator'},
                            {'type': 'string', 'name': 'Grid'},
                            {'type': 'string', 'name': 'Block'},
                            {'type': 'number', 'name': 'Register Per Thread'},
                            {'type': 'number', 'name': 'Shared Memory'},
                            {'type': 'string', 'name': 'Kernel Uses Tensor Cores',
                             'tooltip': consts.TOOLTIP_KERNEL_USES_TC},
                            {'type': 'string', 'name': 'Op is Tensor Cores eligible',
                             'tooltip': consts.TOOLTIP_KERNEL_OP_TC_ELIGIBLE}]
        col_names = ['Calls', 'Total Duration (us)', 'Mean Duration (us)', 'Max Duration (us)', 'Min Duration (us)']
        for column in col_names:
            table['columns'].append({'type': 'number', 'name': column})
        gpu_metrics_columns = self.profile_data.gpu_metrics_parser.get_gpu_metrics_columns()
        table['columns'].extend(gpu_metrics_columns)

        table['rows'] = []
        kernel_list: List[KernelAggByNameOp] = sorted(
            self.profile_data.kernel_list_groupby_name_op, key=lambda x: x.total_duration, reverse=True)
        for agg_by_name_op in kernel_list:
            kernel_op_row = [agg_by_name_op.name, agg_by_name_op.op_name,
                             str(agg_by_name_op.grid), str(agg_by_name_op.block),
                             str(agg_by_name_op.regs_per_thread or '0'), str(agg_by_name_op.shared_memory or '0'),
                             'Yes' if agg_by_name_op.tc_used else 'No',
                             'Yes' if agg_by_name_op.op_tc_eligible else 'No',
                             agg_by_name_op.calls,
                             agg_by_name_op.total_duration, round(agg_by_name_op.avg_duration),
                             agg_by_name_op.max_duration, agg_by_name_op.min_duration]
            if self.profile_data.gpu_metrics_parser.has_blocks_per_sm:
                kernel_op_row.append(round(agg_by_name_op.avg_blocks_per_sm, 2))
            if self.profile_data.gpu_metrics_parser.has_occupancy:
                kernel_op_row.append(round(agg_by_name_op.avg_occupancy, 2))
            table['rows'].append(kernel_op_row)
        return result

    def _generate_kernel_pie_gpu(self):
        pie = {'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}], 'rows': []}
        for _id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            pie['rows'].append([name, row['sum']])
        data = {'total': pie, 'device_target': self.device_target}
        return data

    def _generate_kernel_table_gpu(self):
        table = {}
        result = {
            'metadata': {
                'sort': 'Total Duration (us)'
            },
            'data': table
        }
        table['columns'] = [{'type': 'string', 'name': 'Name'},
                            {'type': 'string', 'name': 'Tensor Cores Used',
                             'tooltip': consts.TOOLTIP_KERNEL_USES_TC}]
        columns = ['count', 'sum', 'mean', 'max', 'min']
        round_digits = [0, 0, 0, 0, 0]
        if self.profile_data.gpu_metrics_parser.has_blocks_per_sm:
            columns.append('blocks_per_sm')
            round_digits.append(2)
        if self.profile_data.gpu_metrics_parser.has_occupancy:
            columns.append('occupancy')
            round_digits.append(2)
        col_names = ['Calls', 'Total Duration (us)', 'Mean Duration (us)', 'Max Duration (us)', 'Min Duration (us)']
        for column in col_names:
            table['columns'].append({'type': 'number', 'name': column})
        gpu_metrics_columns = self.profile_data.gpu_metrics_parser.get_gpu_metrics_columns()
        table['columns'].extend(gpu_metrics_columns)

        table['rows'] = []
        for _id, (name, row) in enumerate(self.profile_data.kernel_stat.iterrows()):
            kernel_row = [name, 'Yes' if row['tc_used'] else 'No']
            for i, column in enumerate(columns):
                kernel_row.append(round(row[column]) if round_digits[i] == 0
                                  else round(row[column], round_digits[i]))
            table['rows'].append(kernel_row)
        return result

    def _generate_tc_pie_gpu(self):
        pie = {'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}], 'rows': []}
        pie['rows'].append(['Using Tensor Cores', self.profile_data.tc_used_ratio])
        pie['rows'].append(['Not Using Tensor Cores', 1.0 - self.profile_data.tc_used_ratio])
        data = {'total': pie}
        return data

    def _generate_kernel_op_table_npu(self):
        table = {}
        result = {
            'metadata': {
                'sort': 'Calls'
            },
            'data': table
        }
        table['columns'] = [{'type': 'string', 'name': 'Name'},
                            {'type': 'number', 'name': 'Calls'},
                            {'type': 'number', 'name': 'Total Durations(us)'},
                            {'type': 'number', 'name': 'Min Durations(us)'},
                            {'type': 'number', 'name': 'Avg Durations(us)'},
                            {'type': 'number', 'name': 'Max Durations(us)'}]
        table['rows'] = []
        for key, value in self.statistic_data.items():
            temp = [key]
            for val in value.values():
                temp.append(val)
            table['rows'].append(temp)
        return result

    def _generate_kernel_pie_npu(self):
        pie = {'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}], 'rows': []}
        for key, val in self.statistic_data.items():
            data = [key, float(val['Total'])]
            pie['rows'].append(data)
        datas = {'total': pie, 'device_target': self.device_target}
        return datas

    def _generate_kernel_table_npu(self):
        table = {'columns': [], 'rows': []}
        result = {
            'metadata': {
                'sort': 'Duration (us)'
            },
            'data': table
        }
        path = self.profile_data.kernel_file_path
        datas = RunGenerator._get_csv_data(path)
        required_column_idxs = {
            'Name': -1,
            'Duration(us)': -1,
            'Accelerator Core': -1
        }
        (name_idx, duration_idx, core_type_idx), column_exist_count = \
            RunGenerator._check_csv_columns(datas[0], required_column_idxs)
        if column_exist_count < 3:
            logger.error('Required column is missing in file "kernel_details.csv"')
        else:
            for column in datas[0]:
                if column in ('Duration(us)', 'Start Time(us)', 'Wait Time(us)', 'Block Dim'):
                    table['columns'].append({'type': 'number', 'name': column})
                else:
                    table['columns'].append({'type': 'string', 'name': column})

            self._handle_kernel_table_rows(name_idx, duration_idx, core_type_idx, datas[1:])
        table['rows'] = datas[1:]
        return result

    @staticmethod
    def _get_csv_data(path: str):
        if path is None:
            return []
        datas = []
        with open(path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                datas.append(row)
        return datas

    def _generate_tc_pie_npu(self):
        pie = {'columns': [{'type': 'string', 'name': 'name'}, {'type': 'number', 'name': 'value'}], 'rows': []}
        for key, val in self.accelerator_data.items():
            pie['rows'].append(['Using ' + key.replace('_', ' '), val])
        data = {'total': pie}
        return data

    @staticmethod
    def _get_gpu_info(device_props, gpu_id):
        if (device_props is None) or (gpu_id >= len(device_props)) or (gpu_id < 0):
            return None

        device_prop: Dict = device_props[gpu_id]
        gpu_info = {}
        name = device_prop.get('name')
        if name is not None:
            gpu_info['Name'] = name

        mem = device_prop.get('totalGlobalMem')
        if mem is not None:
            gpu_info['Memory'] = '{} GB'.format(round(float(mem) / 1024 / 1024 / 1024, 2))
            gpu_info['Memory Raw'] = mem

        major = device_prop.get('computeMajor')
        minor = device_prop.get('computeMinor')
        if major is not None and minor is not None:
            gpu_info['Compute Capability'] = '{}.{}'.format(major, minor)

        return gpu_info

    def _handle_kernel_table_rows(self, name_idx, duration_idx, core_type_idx, rows):
        for row in rows:
            call_name = row[name_idx]
            call_duration = float(row[duration_idx])
            call_type = row[core_type_idx]
            if self.accelerator_data.get(call_type) is not None:
                self.accelerator_data[call_type] += call_duration
            else:
                self.accelerator_data[call_type] = call_duration

            if self.statistic_data.get(call_name) is not None:
                temp = self.statistic_data[call_name]
                temp['Max'] = max(temp['Max'], call_duration)
                temp['Min'] = min(temp['Min'], call_duration)
                temp['Total'] = round(temp['Total'] + call_duration, 2)
                temp['Calls'] += 1
                temp['Average'] = round(temp['Total'] / temp['Calls'], 2)
            else:
                self.statistic_data[call_name] = {
                    'Calls': 1,
                    'Total': call_duration,
                    'Min': call_duration,
                    'Average': call_duration,
                    'Max': call_duration
                }


class DistributedRunGenerator(object):
    def __init__(self, all_profile_data: Iterable[DistributedRunProfileData], span, device_target):
        self.all_profile_data = all_profile_data
        self.span = span
        self.device_target = device_target

    def generate_run_profile(self):
        profile_run = DistributedRunProfile(self.span)
        profile_run.views.append(consts.DISTRIBUTED_VIEW)
        profile_run.gpu_info = self._generate_gpu_info()
        profile_run.steps_to_overlap = self._generate_overlap_graph()
        profile_run.steps_to_wait = self._generate_wait_graph()
        profile_run.comm_ops = self._generate_ops_table()
        return profile_run

    def _generate_gpu_info(self):
        # first key is node name, the second key is process id, the third key is GPU0/,
        # the value is the gpu info json
        result: Dict[str, Dict[str, Dict[str, Dict]]] = OrderedDict()
        index = 0
        for data in sorted(self.all_profile_data, key=lambda x: x.worker):
            if not data.device_props:
                continue

            match = consts.NODE_PROCESS_PATTERN.match(data.worker)
            if match:
                node = match.group(1)
                process_id = match.group(2)
            else:
                logger.warning('cannot parse node name from worker name {}'.format(data.worker))
                node = data.worker
                process_id = index
                index += 1
            if node not in result:
                result[node] = OrderedDict()

            process_id = 'Process ' + str(process_id)
            result[node][process_id] = OrderedDict()
            for used_device in data.used_devices:
                gpu_info = RunGenerator._get_gpu_info(data.device_props, used_device)
                if gpu_info is not None:
                    result[node][process_id]['GPU' + str(used_device)] = gpu_info

        if result:
            for k, v in result.items():
                result[k] = OrderedDict(sorted(v.items()))
            return {
                'metadata': {'title': 'Device Information'},
                'data': result
            }
        else:
            return None

    def _generate_overlap_graph(self):
        result = dict()
        result['metadata'] = {
            'title': 'Computation/Communication Overview',
            'legends': ['Computation', 'Overlapping', 'Communication', 'Other'],
            'units': 'us'
        }
        steps_to_overlap: Dict[str, Dict[str, List[int]]] = OrderedDict()
        steps_to_overlap['all'] = OrderedDict()
        for data in self.all_profile_data:
            steps_to_overlap['all'][data.worker] = [0, 0, 0, 0]
            step_number = len(data.steps_names)
            if step_number <= 0:
                return None
            if self.device_target != 'Ascend':
                DistributedRunGenerator._get_gpu_overlap_data(data, steps_to_overlap)
            else:
                DistributedRunGenerator._get_npu_overlap_data(data, steps_to_overlap)

            steps_to_overlap['all'][data.worker] = [int(x / step_number) for x in steps_to_overlap['all'][data.worker]]
        for k, v in steps_to_overlap.items():
            steps_to_overlap[k] = OrderedDict(sorted(v.items()))
        result['data'] = steps_to_overlap
        return result

    @staticmethod
    def _get_gpu_overlap_data(data, steps_to_overlap):
        for i, step_name in enumerate(data.steps_names):
            steps_to_overlap.setdefault(step_name, OrderedDict())
            costs = data.comm_overlap_costs[i]
            steps_to_overlap[step_name][data.worker] = [
                round(costs.computation - costs.overlap, 3),
                round(costs.overlap, 3),
                round(costs.communication - costs.overlap, 3),
                round(costs.other, 3)
            ]
            steps_to_overlap['all'][data.worker] = [
                sum(x) for x in zip(steps_to_overlap['all'][data.worker], steps_to_overlap[step_name][data.worker])]

    @staticmethod
    def _get_npu_overlap_data(data, steps_to_overlap):
        steps = data.step_to_overlap
        for k, v in steps.items():
            steps_to_overlap.setdefault(k, OrderedDict())
            # v: computation / overlap / communication_not_overlap / free time
            # steps_to_overlap: computation_not_overlap / overlap / communication_not_overlap / other
            steps_to_overlap[k][data.worker] = list(
                [round(v[0] - v[1], 3), round(v[1], 3), round(v[2], 3), round(v[3], 3)])
            steps_to_overlap['all'][data.worker] = [
                sum(x) for x in zip(steps_to_overlap['all'][data.worker], steps_to_overlap[k][data.worker])]

    @staticmethod
    def _get_npu_wait_data(data, steps_to_wait):
        step_number = len(data.step_to_wait)
        if step_number <= 0:
            return
        steps = data.step_to_wait
        for k, v in steps.items():
            steps_to_wait.setdefault(k, OrderedDict())

            trans = round(v.get('trans') * 1000, 3)  # 1ms = 1000us
            wait = round(v.get('Synchronize') * 1000, 3)  # 1ms = 1000us
            steps_to_wait[k][data.worker] = list([trans, wait])
            steps_to_wait['all'][data.worker] = [
                sum(x) for x in zip(steps_to_wait['all'][data.worker], steps_to_wait[k][data.worker])]
        steps_to_wait['all'][data.worker] = [x / step_number for x in steps_to_wait['all'][data.worker]]

    @staticmethod
    def _get_gpu_wait_data(data, steps_to_wait):
        step_number = len(data.step_comm_stats.values())
        if step_number <= 0:
            return
        for step, comm_stats in data.step_comm_stats.items():
            steps_to_wait.setdefault(step, OrderedDict())[data.worker] = [
                round(comm_stats[1], 3),
                round(comm_stats[0] - comm_stats[1], 3)
            ]
            steps_to_wait['all'][data.worker] = [
                sum(x) for x in zip(steps_to_wait['all'][data.worker], steps_to_wait[step][data.worker])]
        steps_to_wait['all'][data.worker] = [int(x / step_number) for x in steps_to_wait['all'][data.worker]]

    def _generate_wait_graph(self):
        result = dict()
        result['metadata'] = {
            'title': 'Synchronizing/Communication Overview',
            'legends': ['Data Transfer Time', 'Synchronizing Time'],
            'units': 'us'
        }
        steps_to_wait: Dict[str, Dict[str, List[int]]] = OrderedDict()

        steps_to_wait['all'] = OrderedDict()
        for data in self.all_profile_data:
            steps_to_wait['all'][data.worker] = [0, 0]
            if self.device_target != 'Ascend':
                DistributedRunGenerator._get_gpu_wait_data(data, steps_to_wait)
            else:
                DistributedRunGenerator._get_npu_wait_data(data, steps_to_wait)
        for k, v in steps_to_wait.items():
            steps_to_wait[k] = OrderedDict(sorted(v.items()))
        result['data'] = steps_to_wait
        return result

    def _generate_ops_table(self):
        result = dict()
        result['metadata'] = {'title': 'Communication Operations Stats'}
        workers_to_comm_ops = OrderedDict()
        # Ignore the span for distributed view
        for data in self.all_profile_data:
            table = {}
            if self.device_target != 'Ascend':
                DistributedRunGenerator._get_gpu_table(data, table)
            else:
                DistributedRunGenerator._get_npu_table(data, table)
            workers_to_comm_ops[data.worker] = table
        result['data'] = OrderedDict(sorted(workers_to_comm_ops.items()))
        return result

    @staticmethod
    def _get_npu_table(data, table):
        table['columns'] = [{'type': 'string', 'name': 'Name'}]
        col_names = [
            'Calls',
            'Total Transit Size (bytes)',
            'Avg Transit Size (bytes)',
            'Elapse Time (us)',
            'Avg Elapse Time (us)',
            'Transit Time (us)',
            'Avg Transit Time (us)'
        ]
        for column in col_names:
            table['columns'].append({'type': 'number', 'name': column})
        table['rows'] = []
        ops = data.comm_op
        for op, stats in ops.items():
            row = [
                op,
                stats[0],
                round(stats[1] * 1024 * 1024, 3),
                round(stats[1] * 1024 * 1024 / stats[0] if stats != 0 else 0),  # 1MB = 1024 * 1024 bytes
                round(stats[2] * 1000, 3),
                round(stats[2] * 1000 / stats[0] if stats != 0 else 0),  # 1ms = 1000us
                round(stats[3] * 1000, 3),
                round(stats[3] * 1000 / stats[0] if stats != 0 else 0)  # 1ms = 1000us
            ]
            table['rows'].append(row)

    @staticmethod
    def _get_gpu_table(data, table):
        table['columns'] = [{'type': 'string', 'name': 'Name'}]
        col_names = [
            'Calls',
            'Total Size (bytes)',
            'Avg Size (bytes)',
            'Total Latency (us)',
            'Avg Latency (us)',
            'Data Transfer Time (us)',
            'Avg Data Transfer Time (us)'
        ]
        for column in col_names:
            table['columns'].append({'type': 'number', 'name': column})
        table['rows'] = []
        for op, stats in data.total_comm_stats.items():
            row = [
                op,
                stats[0],
                round(stats[1], 3),
                round(stats[1] / stats[0] if stats != 0 else 0),
                round(stats[2], 3),
                round(stats[2] / stats[0] if stats != 0 else 0),
                round(stats[3], 3),
                round(stats[3] / stats[0] if stats != 0 else 0)
            ]
            table['rows'].append(row)
