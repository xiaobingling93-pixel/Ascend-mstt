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

import { isEmpty } from 'lodash';
import { fetchPbTxt } from '../../../tf_graph_common/parser';
import { NPU_PREFIX, BENCH_PREFIX } from '../../../tf_graph_common/common';
import { safeJSONParse } from '../../../utils';

export interface UseMatchedType {
  saveMatchedNodesLink: (selection: any) => Promise<any>;
  addMatchedNodesLink: (
    npuNodeName: string,
    benchNodeName: string,
    selection: any,
    renderHierarchy: any,
  ) => Promise<any>;
  queryMatchedStateList: (selection: any) => Promise<any>;
  deleteMatchedNodesLink: (
    npuNodeName: string,
    benchNodeName: string,
    selection: any,
    renderHierarchy: any,
  ) => Promise<any>;
}

const useMatched = (): UseMatchedType => {
  const requestAddMatchNodes = async (npuNodeName: string, benchNodeName: string, metaData: any): Promise<any> => {
    const params = new URLSearchParams();
    params.set('npuNodeName', JSON.stringify(npuNodeName));
    params.set('benchNodeName', JSON.stringify(benchNodeName));
    params.set('metaData', JSON.stringify(metaData));
    // 接口请求
    const precisionPath = `addMatchNodes?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    // 接口返回
    const mactchResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return mactchResult;
  };

  const requestDeleteMatchNodes = async (npuNodeName: string, benchNodeName: string, metaData: any): Promise<any> => {
    const params = new URLSearchParams();
    params.set('npuNodeName', JSON.stringify(npuNodeName));
    params.set('benchNodeName', JSON.stringify(benchNodeName));
    params.set('metaData', JSON.stringify(metaData));
    // 接口请求
    const precisionPath = `deleteMatchNodes?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    // 接口返回
    const mactchResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return mactchResult;
  };

  const requestMatchStateList = async (metaData: any): Promise<any> => {
    const params = new URLSearchParams();
    params.set('metaData', JSON.stringify(metaData));
    // 接口请求
    const precisionPath = `getMatchedStateList?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    // 接口返回
    const mactchResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return mactchResult;
  };

  const queryMatchedStateList = async (selection: any): Promise<any> => {
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const matchStateList = await requestMatchStateList(metaData);
    return matchStateList;
  };

  const saveMatchedNodesLink = async (selection: any): Promise<any> => {
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const params = new URLSearchParams();
    params.set('metaData', JSON.stringify(metaData));
    // 接口请求
    const precisionPath = `saveData?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    // 接口返回
    const saveResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return saveResult;
  };

  const addMatchedNodesLink = async (
    npuNodeName: string,
    benchNodeName: string,
    selection: any,
    renderHierarchy: any,
  ): Promise<any> => {
    if (isEmpty(npuNodeName) || isEmpty(benchNodeName)) {
      return {
        success: false,
        error:'调试侧节点或标杆节点为空'
      };
    }
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const matchResult = await requestAddMatchNodes(npuNodeName, benchNodeName, metaData);
    if (matchResult.success) {
      const graphNpuNodeName = NPU_PREFIX + npuNodeName;
      const graphBenchNodeName = BENCH_PREFIX + benchNodeName;
      const graphNpuNodeInputData = renderHierarchy.npu?.index?.[graphNpuNodeName]?.node?.inputData;
      const graphNpuNodeOutputData = renderHierarchy.npu?.index?.[graphNpuNodeName]?.node?.outputData;
      const intputStatisticalDiff = matchResult.data.intput_statistical_diff;
      const outputStatisticalDiff = matchResult.data.output_statistical_diff;
      // 更新节点之间的匹配关系
      updateGraphNodeData(graphNpuNodeInputData, intputStatisticalDiff);
      updateGraphNodeData(graphNpuNodeOutputData, outputStatisticalDiff);
      renderHierarchy.npu.index[graphNpuNodeName].node.matchedNodeLink = [graphBenchNodeName];
      renderHierarchy.bench.index[graphBenchNodeName].node.matchedNodeLink = [graphNpuNodeName];
      renderHierarchy.npu.index[graphNpuNodeName].node.nodeAttributes._linked_node = [graphBenchNodeName];
      renderHierarchy.bench.index[graphBenchNodeName].node.nodeAttributes._linked_node = [graphNpuNodeName];
      // 更新匹配精度,节点重新上色
      const precisionIndex = matchResult.data.precision_error;
      const nodeAtts = renderHierarchy.npu.index[graphNpuNodeName].node.attr;
      const precisionIndexObj = nodeAtts?.find((item) => item.key === 'precision_index');
      if (precisionIndexObj) {
        precisionIndexObj.value = precisionIndex;
      } else {
        nodeAtts.push({
          key: 'precision_index',
          value: precisionIndex,
        });
      }
    }
    return matchResult;
  };

  const deleteMatchedNodesLink = async (
    npuNodeName: string,
    benchNodeName: string,
    selection: any,
    renderHierarchy: any,
  ): Promise<any> => {
    if (isEmpty(npuNodeName) || isEmpty(benchNodeName)) {
      return {
        success: false,
        error:'调试侧节点或标杆节点为空'
      };
    }
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const matchResult = await requestDeleteMatchNodes(npuNodeName, benchNodeName, metaData);
    matchResult.success = true;
    if (matchResult.success) {
      const graphNpuNodeName = NPU_PREFIX + npuNodeName;
      const graphBenchNodeName = BENCH_PREFIX + benchNodeName;
      const graphNpuNodeInputData = renderHierarchy.npu?.index?.[graphNpuNodeName]?.node?.inputData;
      const graphNpuNodeOutputData = renderHierarchy.npu?.index?.[graphNpuNodeName]?.node?.outputData;
      // 清空节点之间的匹配关系
      deleteMatchedNodeData(graphNpuNodeInputData);
      deleteMatchedNodeData(graphNpuNodeOutputData);
      renderHierarchy.npu.index[graphNpuNodeName].node.matchedNodeLink = [];
      renderHierarchy.bench.index[graphBenchNodeName].node.matchedNodeLink = [];
      renderHierarchy.npu.index[graphNpuNodeName].node.nodeAttributes._linked_node = [];
      renderHierarchy.bench.index[graphBenchNodeName].node.nodeAttributes._linked_node = [];
      // 更新匹配精度,节点重新上色
      const nodeAtts = renderHierarchy.npu.index[graphNpuNodeName].node.attr;
      const precisionIndexObj = nodeAtts?.filter((item) => item.key === 'precision_index');
      renderHierarchy.npu.index[graphNpuNodeName].node.attr = precisionIndexObj;
    }
    return matchResult;
  };

  const updateGraphNodeData = (graphNpuNodeData, statisticalDiff): void => {
    if (isEmpty(statisticalDiff) || isEmpty(graphNpuNodeData)) {
      return;
    }
    for (const key in statisticalDiff) {
      if (Object.prototype.hasOwnProperty.call(statisticalDiff, key)) {
        const value = statisticalDiff[key];
        graphNpuNodeData[key] = {
          ...graphNpuNodeData[key], // 如果 graphNpuNodeData[key] 可能为 undefined，则需要额外处理以避免错误
          ...value,
        };
      }
    }
  };
  const deleteMatchedNodeData = (graphNpuNodeData): void => {
    const keysToRemove = [
      'MaxAbsErr',
      'MinAbsErr',
      'NormAbsErr',
      'MeanAbsErr',
      'MaxRelativeErr',
      'MinRelativeErr',
      'NormRelativeErr',
      'MeanRelativeErr',
    ];
    for (const key in graphNpuNodeData) {
      if (Object.prototype.hasOwnProperty.call(graphNpuNodeData, key)) {
        const fildObj = graphNpuNodeData[key];
        keysToRemove.forEach((keyToRemove) => {
          // 确保要删除的键存在于当前对象中
          delete fildObj[keyToRemove];
        });
      }
    }
  };

  return {
    saveMatchedNodesLink,
    addMatchedNodesLink,
    queryMatchedStateList,
    deleteMatchedNodesLink,
  };
};

export default useMatched;
