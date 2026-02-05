# Copyright (c) 2025, Huawei Technologies Co., Ltd.
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
import os
import csv


def main():
    all_data_size_mb = []
    for i in range(11):
        data_size_mb = 2 ** i
        all_data_size_mb.append(data_size_mb)
    all_bandwidth_gb_per_second = [
        3, 6, 9, 12, 15, 17, 18, 18, 18, 19, 19
    ]
    result_file = os.environ.get("FILE_NAME", "p2p_intra_node.csv")
    with open(result_file, "a+") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(all_data_size_mb)
        f_csv.writerow(all_bandwidth_gb_per_second)


if __name__ == "__main__":
    main()
