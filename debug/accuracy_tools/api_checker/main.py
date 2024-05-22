import argparse
import json
import os
from common.logger import logger


def main():
    parser = argparse.ArgumentParser(description='Generate JSON based on user input')
    
    parser.add_argument('--mode', type=int, default=0, help='Dump mode (0 for all data and 1 for select api)')
    parser.add_argument('--out', type=str, default="./data", help='Dump data output dir')
    parser.add_argument('--net_name', type=str, default="MyNet", help='Network name (e.g., MyNet)')
    parser.add_argument('--iteration', type=str, default="0", help='Iteration range (e.g., "0|5-8|100-120")')
    parser.add_argument('--saved_data', type=str, default="tensor", help='Saved data type ("tensor" for dump tensor and "statistic" for statistic data)')
    parser.add_argument('--input_output', type=int, default="0", help='Input output flag (0 for all and 1 for input and 2 for output)')
    parser.add_argument('--kernels', type=str, nargs='+', default="", help='List of selected kernels only valid when mode is 1(e.g., "Default/Conv-op12")')
    parser.add_argument('--support_device', type=int, nargs='+', required=True, help='List of supported devices (e.g., 0 1 2 3 4 5 6 7)')
    parser.add_argument('--e2e_enable', type=bool, default=True, help='Enable end-to-end dump (true/false)')
    parser.add_argument('--e2e_trans_flag', type=bool, default=True, help='End-to-end trans flag (true/false)')
    parser.add_argument('--output_dir', type=str, default="./", help='Output JSON file path')

    args = parser.parse_args()

    json_data = {
        "common_dump_settings": {
            "dump_mode": args.mode,
            "path": os.path.realpath(args.out),
            "net_name": args.net_name,
            "iteration": args.iteration,
            "saved_data": args.saved_data,
            "input_output": args.input_output,
            "kernels": list(args.kernels),
            "support_device": list(args.support_device),
            "op_debug_mode": 0,
            "file_format": "npy"
        },
        "e2e_dump_settings": {
            "enable": args.e2e_enable,
            "trans_flag": args.e2e_trans_flag,
            "save_kernel_args": True
        }
    }
    dump_json_path = os.path.realpath(args.output_dir)
    output_json_path = os.path.join(args.output_dir, "dump.json")
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    logger.info(f"JSON data saved to {output_json_path}")
if __name__ == '__main__':
    main()
