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
import { fetchPbTxt, safeJSONParse } from '../../utils';

const useNodeInfoDomain = (): { getMatchNodeInfo: (nodeInfo: any, metaData: any) => Promise<any> } => {
  const getMatchNodeInfo = async (nodeInfo: any, metaData: any): Promise<any> => {
    const params = new URLSearchParams();
    params.set('nodeInfo', JSON.stringify(nodeInfo));
    params.set('metaData', JSON.stringify(metaData));
    // 接口请求
    const precisionPath = `getNodeInfo?${String(params)}`;
    const precisionStr = await fetchPbTxt(precisionPath); // 获取异步的 ArrayBuffer
    const decoder = new TextDecoder();
    const decodedStr = decoder.decode(precisionStr); // 解码 ArrayBuffer 到字符串
    // 接口返回
    const mactchResult = safeJSONParse(decodedStr.replace(/"None"/g, '{}'));
    return mactchResult;
  };

  return {
    getMatchNodeInfo,
  };
};

export default useNodeInfoDomain;
