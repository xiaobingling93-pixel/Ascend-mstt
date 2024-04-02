# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import importlib
import torch.multiprocessing as mp


def check_and_convert_weight(args):
    import torch
    from transformers import AutoModelForCausalLM
    try:
        output_mg2hg_path = os.path.join(args.output_model_dir, 'mg2hg')
        hf_model = AutoModelForCausalLM.from_pretrained(
            output_mg2hg_path, device_map="cpu", torch_dtype=torch.float16)
        hf_model.save_pretrained(output_mg2hg_path, safe_serialization=True)
    except ModuleNotFoundError as e:
        print('failed to convert bin 2 safetensors')
        raise exc from e


def load_model(model_name):
    module_name = f"{model_name}"
    try:
        converter = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise exc from e
    return converter


def main():
    parser = argparse.ArgumentParser(
        description="convert weight to huggingface format")

    parser.add_argument('-m', '--model', type=str, required=True,
                        choices=['llama', 'bloom', 'gptneox'],
                        help='Type of the model')
    parser.add_argument('-i', '--input-model-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('-o', '--output-model-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')
    parser.add_argument('--convert-to-safetensors', action='store_false',
                        help='convert .bin to safetensors')

    known_args, _ = parser.parse_known_args()
    loader = importlib.import_module('load_utils')
    saver = load_model(known_args.model)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=50)

    print("Starting saver...")
    saver_proc = mp.Process(
        target=saver.save_model_checkpoint, args=(queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()

    if args.convert_to_safetensors:
        print("converting .bin to safetensors...")
        check_and_convert_weight(args)

    print("Done!")


if __name__ == '__main__':
    main()
