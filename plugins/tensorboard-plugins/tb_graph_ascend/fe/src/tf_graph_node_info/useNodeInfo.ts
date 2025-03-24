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
import { isEmpty, cloneDeep } from 'lodash';
import useNodeInfoDomain from './domain/useNodeInfoDomain';
import type { MatchNodeInfo } from './type';
import type { MactchResult } from '../types/service';
import { BENCH_PREFIX, NPU_PREFIX } from '../tf_graph_common/common';
import { safeJSONParse } from '../utils';

export interface UseNodeInfoType {
  getNodeInfo: (
    nodeInfo: {
      nodeName: string;
      nodeType: string;
    },
    metaData: any,
  ) => Promise<MatchNodeInfo>;
  getIoDataSet: (
    npuNode: any,
    benchNode: any,
    type: 'inputData' | 'outputData',
  ) => {
    matchedIoDataset: Array<Record<string, unknown>>;
    unMatchedNpuIoDataset: Array<Record<string, unknown>>;
    unMatchedBenchIoDataset: Array<Record<string, unknown>>;
  };
  getDetailDataSet: (npuNode: any, benchNode: any) => Array<Record<string, unknown>>;
  converMatchArrayToObject: (resArray: Array<any>) => Array<MactchResult['result']>;
}

const useNodeInfo = (): UseNodeInfoType => {
  const useNodeInfoService = useNodeInfoDomain();
  /**
   * 获取节点信息
   * @param node_name
   * @param graph_name
   * @param run_name
   * @returns
   */
  const getNodeInfo = async (
    nodeInfo: { nodeName: string; nodeType: string },
    metaData: any,
  ): Promise<MatchNodeInfo> => {
    if (sessionStorage.getItem(JSON.stringify(nodeInfo))) {
      // 缓存中存在
      return JSON.parse(sessionStorage.getItem(JSON.stringify(nodeInfo)) || '') as MatchNodeInfo;
    }
    const mactchResult = await useNodeInfoService.getMatchNodeInfo(nodeInfo, metaData);
    mactchResult.data = convertNodeInfo(mactchResult.data); // 提取有效数据，统一命名
    sessionStorage.setItem(JSON.stringify(nodeInfo), JSON.stringify(mactchResult)); // 缓存
    return mactchResult;
  };

  const convertNodeInfo = (nodeInfo: any): MatchNodeInfo['data'] => {
    if (isEmpty(nodeInfo)) {
      return {};
    }
    return {
      name: nodeInfo.id,
      inputData: nodeInfo.input_data,
      outputData: nodeInfo.output_data,
      stackData: !isEmpty(nodeInfo.stack_info) ? JSON.stringify(nodeInfo.stack_info) : '',
      suggestions: nodeInfo.suggestions,
      subnodes: nodeInfo.subnodes,
      include: nodeInfo.subnodes?.length,
    };
  };

  /**
   * 将匹配结果数组转换为对象
   * @param resArray 
   * @example
   * [request.args.get("NPU"),
   *  max_precision_index,
   * [
   *  [
          ['npu_key', npu_key],
          ['Max diff', NPU_max - Bench_max],
          ['Min diff', NPU_min - Bench_min],
          ['Mean diff', NPU_mean - Bench_mean],
          ['L2norm diff', NPU_norm - Bench_norm],
          ['MaxRelativeErr', max_relative_err],
          ['MinRelativeErr', min_relative_err],
          ['MeanRelativeErr', mean_relative_err],
          ['NormRelativeErr', norm_relative_err]
      ]
        .....
      ],
   *   output(同上)]
   * @returns
   * {
   *  NPU_node_name: string,
   *  max_precision_error: number,
   *  input: {inputargs:{
   *    Max diff: number,
   *    Min diff: number,
   *    Mean diff: number,
   * },{}}
   * output: [{
   *    Max diff: number,
   *    Min diff: number,
   *    Mean diff: number,
   * }]
   */
  const converMatchArrayToObject = (resArray: Array<any>): Array<MactchResult['result']> => {
    if (resArray.length === 0) {
      return [];
    }
    //内部函数，将二维数组转换为对象
    const _covertIo = (arrayData: Array<any>): Record<string, unknown> | undefined => {
      if (isEmpty(arrayData)) {
        return undefined;
      }
      const inputItems = {};
      arrayData.forEach((inputArray: any) => {
        const inputKey = inputArray[0];
        const inputValue = {};
        inputArray.slice(1).map((item: any) => {
          inputValue[item[0]] = item[1];
        });
        inputItems[inputKey] = inputValue;
      });
      return inputItems;
    };

    // 将数组转换为对象
    const matchedeDataset = resArray.map((item) => {
      return {
        NPU_node_name: item?.[0],
        max_precision_error: item?.[1],
        inputData: _covertIo(item?.[2]),
        outputData: _covertIo(item?.[3]),
      };
    });

    return matchedeDataset as Array<MactchResult['result']>;
  };

  /**
   * 获取匹配的输入输出数据
   * @param npuNode NPU节点信息
   * @param benchNode 匹配的节点信息
   * @param name 'inputData' | 'outputData'
   * @returns { matchedDataset: Array<{}>, unMatchedNpuDataset: Array<{}>, unMatchedBenchDataset: Array<{}> }
   */
  const getIoDataSet = (
    npuNode: any,
    benchNode: any,
    type: 'inputData' | 'outputData',
  ): {
    matchedIoDataset: Array<Record<string, unknown>>;
    unMatchedNpuIoDataset: Array<Record<string, unknown>>;
    unMatchedBenchIoDataset: Array<Record<string, unknown>>;
  } => {
    if (isEmpty(npuNode?.[type]) && isEmpty(benchNode?.[type])) {
      return {
        matchedIoDataset: [],
        unMatchedNpuIoDataset: [],
        unMatchedBenchIoDataset: [],
      };
    }
    const npuNodeName = npuNode?.name.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    const benchNodeName = benchNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '');
    const npuData = cloneDeep(npuNode?.[type]); // 获取当前节点的输入数据
    const benchData = cloneDeep(benchNode?.[type]); // 获取匹配节点的输入数据

    const matchedIoDataset: Array<Record<string, unknown>> = []; // 初始化输入数据集
    const unMatchedBenchIoDataset: Array<Record<string, unknown>> = [];
    const unMatchedNpuIoDataset: Array<Record<string, unknown>> = [];
    const npuKeys = Object.keys(npuData || {});
    const benchKeys = Object.keys(benchData || {});
    const minLength = Math.min(npuKeys.length, benchKeys.length);
    for (let i = 0; i < minLength; i++) {
      const npuKey = npuKeys[i];
      const benchKey = benchKeys[i];
      matchedIoDataset.push({
        name: npuKey.replace(`${npuNodeName}.`, ''),
        isMatched: true,
        ...npuData[npuKey],
      });
      matchedIoDataset.push({
        name: benchKey.replace(`${benchNodeName}.`, ''),
        isBench: true,
        isMatched: true,
        ...benchData[benchKey],
      });
      delete npuData[npuKey];
      delete benchData[benchKey];
    }
    Object.keys(npuData || {}).forEach((key) => {
      if (npuData[key] !== 'None') {
        unMatchedNpuIoDataset.push({
          name: key.replace(`${npuNodeName}.`, ''),
          ...npuData[key],
        });
      }
    });
    Object.keys(benchData || {}).forEach((key) => {
      if (benchData[key] !== 'None') {
        unMatchedBenchIoDataset.push({
          name: key.replace(`${benchNodeName}.`, ''),
          isBench: true,
          ...benchData[key],
        });
      }
    });
    return { matchedIoDataset, unMatchedNpuIoDataset, unMatchedBenchIoDataset };
  };

  const getDetailDataSet = (npuNode: any, benchNode: any): Array<Record<string, unknown>> => {
    if (isEmpty(npuNode) && isEmpty(benchNode)) {
      return [];
    }
    const nodeName = `NPU节点：${npuNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '')}`;
    const benchNodeName = `标杆节点：${benchNode?.name?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), '')}`;
    const detailData: Array<Record<string, unknown>> = [];
    // 获取stackInfo
    const stackInfo: Record<string, unknown> = {};
    const npustackInfo = npuNode?.stackData;
    const benchstackInfo = benchNode?.stackData;
    const title = 'title';
    if (!isEmpty(npustackInfo)) {
      stackInfo[nodeName] = safeJSONParse(npustackInfo.replace(/'/g, '"'))?.join('\n');
    }
    if (!isEmpty(benchstackInfo)) {
      stackInfo[benchNodeName] = safeJSONParse(benchstackInfo.replace(/'/g, '"'));
    }
    if (!isEmpty(stackInfo)) {
      stackInfo[title] = 'stackInfo';
      detailData.push(stackInfo);
    }
    // 获取suggestions
    const suggestion: Record<string, unknown> = {};
    const npusuggestion = npuNode?.suggestions;
    const benchsuggestion = benchNode?.suggestions;
    if (!isEmpty(npusuggestion)) {
      suggestion[nodeName] = converObjectToString(npusuggestion);
    }
    if (!isEmpty(benchsuggestion)) {
      suggestion[benchNodeName] = converObjectToString(benchsuggestion);
    }
    if (!isEmpty(suggestion)) {
      suggestion[title] = 'suggestions';
      detailData.push(suggestion);
    }
    return detailData;
  };

  const converObjectToString = (obj: any): string => {
    return Object.entries(obj)
      .map(([key, value]) => `${key}: ${value}`)
      .join('\n');
  };

  return {
    getNodeInfo,
    getIoDataSet,
    getDetailDataSet,
    converMatchArrayToObject,
  };
};

export default useNodeInfo;
