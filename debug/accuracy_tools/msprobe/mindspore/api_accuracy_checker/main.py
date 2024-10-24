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

from msprobe.mindspore.api_accuracy_checker.api_accuracy_checker import ApiAccuracyChecker


def api_checker_main(args):
    api_accuracy_checker = ApiAccuracyChecker()
    api_accuracy_checker.parse(args.api_info_file)
    api_accuracy_checker.run_and_compare()
    api_accuracy_checker.to_detail_csv(args.out_path)
    api_accuracy_checker.to_result_csv(args.out_path)
