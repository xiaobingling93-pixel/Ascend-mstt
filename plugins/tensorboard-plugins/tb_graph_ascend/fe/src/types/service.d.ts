/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * /mactch 接口返回值
 * @typedef {Object} mactchResult
 * @property {boolean} success 是否匹配成功
 * @property {string} error 错误信息
 * @property {Object} data 匹配成功的数据
 */
export interface MactchResult {
  success: boolean;
  error?: string;
  data?: [];
  result?: {
    NPU_node_name: string;
    max_precision_error: string;
    inputData: {
      [key: string]: {
        'Max diff': number;
        'Min diff': number;
        'Mean diff': number;
        'L2norm diff': number;
        MaxRelativeErr: number;
        MinRelativeErr: number;
        MeanRelativeErr: number;
        NormRelativeErr: number;
      };
    };
    outputData: {
      [key: string]: {
        'Max diff': number;
        'Min diff': number;
        'Mean diff': number;
        'L2norm diff': number;
        MaxRelativeErr: number;
        MinRelativeErr: number;
        MeanRelativeErr: number;
        NormRelativeErr: number;
      };
    };
    result_array: [];
  };
}
