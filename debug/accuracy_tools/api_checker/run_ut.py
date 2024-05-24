import os
import sys
import json
import importlib
import inspect
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from compare.compare import Comparator
from common.logger import logger
from common.json_parser import convert_json


def _run_ut_parser(parser):
    parser.add_argument(
        "-i", "--input", dest="input_file", required=True, type=str,
        help="<Required> Input josn file containing the API information"
    )
    parser.add_argument(
        "-o", "--output", dest="output_path", required=False, type=str,
        help="<Optional> Output path to store the comparison result"
    )


def get_ops_ut(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and name.startswith("UT"):
            return obj


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def load_npy(file_path):
    return np.load(file_path)


def _run_ut():
    parser = argparse.ArgumentParser()
    _run_ut_parser(parser)
    args = parser.parse_args(sys.argv[1:])
    out_path = os.path.realpath(args.output_path) if args.output_path else "./"
    data_path = os.path.realpath(args.input_file)
    comparator = Comparator(out_path, False)
    cur_path = os.path.dirname(os.path.realpath(__file__))
    api_mapping_path = os.path.join(cur_path, "api_mapping.json")
    
    with open(api_mapping_path, 'r') as f:
        api_mapping_dict = json.load(f)
        
    
    file_groups = defaultdict(lambda: {'json': None, 'args': [], 'output': []})
    
    for dirpath, _, filenames in os.walk(data_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            base_name, ext = os.path.splitext(filename)
            if ext == ".csv":
                continue
            real_name = base_name
            if real_name.isdigit():
                mapping_csv_path = os.path.join(dirpath, "mapping.csv")
                mapping_df = pd.read_csv(mapping_csv_path, header=None, names=['filename', 'realname'])
                mapping_col = mapping_df[mapping_df['filename'] == filename]
                real_name = mapping_col['realname'].tolist()[0]
            
            parts = real_name.split('.')
            op_type = parts[0]
            op_name = parts[1]
            file_groups[op_name]['type'] = op_type
            if ext == ".json":
                file_groups[op_name]['json'] = file_path
            elif ext == ".npy":
                if 'input' in real_name:
                    file_groups[op_name]['args'].append(file_path)
                elif 'output' in real_name:
                    file_groups[op_name]['output'].append(file_path)
    
    for op_name, files in file_groups.items():
        json_path = files['json']
        input_paths = files['args']
        output_paths = files['output']
        op_type = files['type']
        if op_type in api_mapping_dict:
            module_name = "common_ut"
            api_name = op_type + "_" + api_mapping_dict[op_type]
        else:
            module_name = op_type + "_ut"
            api_name = op_name
        
        if not os.path.exists(f"ut_cast/{module_name}.py"):
            logger.warning(f"{op_type} not support compare now")
            continue
    
        if json_path:
            kwargs = convert_json(load_json(json_path))
            args = [load_npy(npy_path) for npy_path in sorted(input_paths)]
            output = [load_npy(npy_path) for npy_path in sorted(output_paths)]
            module = importlib.import_module(f"ut_case.{module_name}")
            if not module:
                logger.warning(f"load {module} failed")
                continue
            
            ops_ut = get_ops_ut(module)
            try:
                ops_ut(
                    api_name,
                    args,
                    kwargs,
                    output,
                    real_data=True,
                    stack=None,
                    comparator=comparator
                ).compare()
            except Exception as e:
                logger.warning(f">>>[{op_name}] Compare failed.Reason: {e}")


if __name__ == '__main__':
    _run_ut()