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
import { fetchPbTxt } from '../../../utils';
import { safeJSONParse } from '../../../utils';
import request from '../../../utils/request';
import { UseMatchedType, MatchResultType } from '../../type';
const useMatched = (): UseMatchedType => {
  const requestAddMatchNodes = async (npuNodeName: string, benchNodeName: string, metaData: any, isMatchChildren: boolean): Promise<any> => {
    const params = {
      npuNodeName: npuNodeName,
      benchNodeName: benchNodeName,
      metaData: JSON.stringify(metaData),
      isMatchChildren
    };
    const mactchResult = await request({ url: 'addMatchNodes', method: 'GET', params: params });
    return mactchResult;
  };

  const requestDeleteMatchNodes = async (npuNodeName: string, benchNodeName: string, metaData: any, isUnMatchChildren: boolean): Promise<any> => {
    const params = {
      npuNodeName: npuNodeName,
      benchNodeName: benchNodeName,
      metaData: JSON.stringify(metaData),
      isUnMatchChildren
    };
    const mactchResult = await request({ url: 'deleteMatchNodes', method: 'GET', params: params });
    return mactchResult;
  };

  const saveMatchedNodesLink = async (selection: any): Promise<any> => {
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const params = new URLSearchParams();
    params.set('metaData', JSON.stringify(metaData));
    const precisionPath = `saveData?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    const saveResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return saveResult;
  };
  const saveMatchedRelations = async (selection: any): Promise<any> => {
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const params = new URLSearchParams();
    params.set('metaData', JSON.stringify(metaData));
    const precisionPath = `saveMatchedRelations?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    const saveResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return saveResult;
  };

  const addMatchedNodesLinkByConfigFile = async (condfigFile: string, selection: any): Promise<MatchResultType> => {
    if (isEmpty(condfigFile)) {
      return {
        success: false,
        error: '请选择配置文件',
      };
    }
    const params = {
      configFile: condfigFile,
      metaData: JSON.stringify(selection),
    };
    const mactchResult = await request({ url: 'addMatchNodesByConfig', method: 'GET', params: params });

    return mactchResult as MatchResultType;
  };

  const addMatchedNodesLink = async (
    npuNodeName: string,
    benchNodeName: string,
    selection: any,
    isMatchChildren: boolean
  ): Promise<MatchResultType> => {
    if (isEmpty(npuNodeName) || isEmpty(benchNodeName)) {
      return {
        success: false,
        error: '调试侧节点或标杆节点为空',
      };
    }
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const matchResult: MatchResultType = await requestAddMatchNodes(npuNodeName, benchNodeName, metaData, isMatchChildren);
    return matchResult;
  };

  const deleteMatchedNodesLink = async (npuNodeName: string, benchNodeName: string, selection: any, isUnMatchChildren: boolean): Promise<any> => {
    if (isEmpty(npuNodeName) || isEmpty(benchNodeName)) {
      return {
        success: false,
        error: '调试侧节点或标杆节点为空',
      };
    }
    const metaData = {
      run: selection.run,
      tag: selection.tag,
    };
    const matchResult: MatchResultType = await requestDeleteMatchNodes(npuNodeName, benchNodeName, metaData, isUnMatchChildren);
    return matchResult;
  };

  return {
    saveMatchedNodesLink,
    addMatchedNodesLink,
    saveMatchedRelations,
    deleteMatchedNodesLink,
    addMatchedNodesLinkByConfigFile,
  };
};

export default useMatched;
