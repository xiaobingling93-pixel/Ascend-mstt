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
import '@vaadin/tabs';
import '@vaadin/tabsheet';
import { Notification } from '@vaadin/notification';
import { isEmpty } from 'lodash';
import { PolymerElement, html } from '@polymer/polymer';
import { observe, customElement, property } from '@polymer/decorators';
import useNodeInfo from './useNodeInfo';
import './components/tf_vaadin_table/index';
import './components/tf_vaddin_text_table/index';
import './components/tf_resize_height/index';
import * as tf_graph_hierarchy from '../tf_graph_common/hierarchy';
import type { UseNodeInfoType } from './useNodeInfo';
import { GroupNode, Metanode, OpNode } from '../tf_graph_common/graph';
import { BENCH_PREFIX, NPU_PREFIX } from '../tf_graph_common/common';

@customElement('tf-graph-vaadin-tab')
class TfGraphNodeInfo extends PolymerElement {
  static readonly template = html`
    <style>
      :host {
        --selected-color: rgb(255, 255, 255); 
        --matched-color: rgb(201, 201, 201);
      }
      .tab-content-wrapper{
        height: 100%;
        display: flex;
        justify-content: space-between;
      }
      .io-vaadin-table {
        width: 100%;
        height: 100%;
      }
      vaadin-tabsheet::part(content){
        background-color: white;
      }
      .vaadin-tabs{
        background-color:white;
      }
      .vaadin-tab{
        font-size: 14px;
      }
      .table-wrapper{
        height: 100%;
        width: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .node-info-wrapper{
        display: flex;
        justify-content: space-between;
        background-color:rgb(237 237 237);
      }
      .node-info{
        font-family: Roboto, sans-serif;
        padding-left: 20px;
        display: flex;
        justify-content: flex-start;
        font-weight: 400;
        font-size: 14px;

      }
      .node-info-item{
        margin-right: 20px;
        display: flex;
        align-items: center;
      }
      .legend-wrapper{
        display: flex;
        align-items: center;
      }
      .legend-selected{
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: var(--selected-color);
        margin-right: 5px;
      }
      .legend-matched{
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: var(--matched-color);
        margin-right: 5px;
      }
      .matched-yes {
          display: inline-block;
          width: 10px;
          height: 8px;
          background: #52c41a;
          padding-top: 2px;
          margin-right: 5px;
          border-radius: 50%;
        }
      .matched-no {
        display: inline-block;
        width: 10px;
        height: 8px;
        background: #ff4d4f;
        padding-top: 2px;
        margin-right: 5px;
        border-radius: 50%;
      }
    </style>
    <vaadin-tabsheet>
      <vaadin-tabs slot="tabs" class='vaadin-tabs'>
        <vaadin-tab id="io-tab" class='vaadin-tab' >
          <template is="dom-if" if="[[!isSingleGraphNode]]">
            比对详情
          </template>
          <template is="dom-if" if="[[isSingleGraphNode]]">
            节点详情
          </template>
        </vaadin-tab>
        <vaadin-tab id="stack-info-tab" class='vaadin-tab'>节点信息</vaadin-tab>
      </vaadin-tabs>
     
      <div tab="io-tab" class='vaadin-tab-content'>
        <tf-resize-height height="{{height}}">   
          <div class='table-wrapper'>
            <div class='node-info-wrapper'>
              <div class='node-info'>
              <template is="dom-if" if="[[npuNodeName]]">
                <p class='node-info-item selected-node'>
                  <span class='legend-selected'></span>
                  目标节点：[[npuNodeName]]
                </p>
                </template>
                <template is="dom-if" if="[[benchNodeName]]">
                  <p class='node-info-item match-node' >
                    <span class='legend-matched'></span>
                    标杆节点：[[benchNodeName]]
                  </p>
                </template>
              </div>
              <template is="dom-if" if="[[!isSingleGraphNode]]">
                <div class='node-info'>
                  <p class='node-info-item '>
                    <span class='matched-yes'></span>
                    已匹配
                  </p>
                  <p class='node-info-item match-node' >
                    <span class='matched-no'></span>
                    未匹配
                  </p>
                </div>
              </template>
            </div>

              <tf-vaadin-table 
                id='main-table' 
                class='io-vaadin-table' 
                io-dataset="[[ioDataset]]"
                tooltips="[[tooltips]]"
                handle-cell-click="[[handleGridCellClick]]"
                is-single-graph-node="[[isSingleGraphNode]]"
              >
              <tf-vaadin-table/>
          </div>
        </tf-resize-height>
      </div>
      <div tab="stack-info-tab" class='vaadin-tab-content'>
        <tf-resize-height height="{{height}}">   
          <div class='table-wrapper'>
            <tf-vaadin-text-table
              id='main-table' 
              class='io-vaadin-table' 
              dataset="[[detailData]]"
              handle-cell-click="[[handleGridCellClick]]"
            >
            <tf-vaadin-text-table/>
          </div>
        </tf-resize-height>
      </div>
    </vaadin-tabsheet>
  `;

  @property({
    type: Object,
    computed: '_getNode(selectedNode)',
  })
  _node: GroupNode | OpNode | undefined | null;

  @property({
    type: Object,
  })
  _matchedNode: any;

  @property({ type: String })
  selectedNode: string = '';

  @property({ type: Object })
  selection: any;

  @property({ type: Object })
  graphHierarchy?: tf_graph_hierarchy.MergedHierarchy;

  @property({ type: Array })
  ioDataset: any[] = [];

  @property({ type: Array })
  detailData: any[] = [];

  @property({ type: String })
  npuNodeName?: string;

  @property({ type: String })
  benchNodeName?: string;

  @property({ type: Number })
  isSingleGraphNode: boolean = false;

  useNodeInfo: UseNodeInfoType = useNodeInfo();

  @observe('_node')
  async _updateTableData(): Promise<void> {
    if (!this._node) {
      this.set('ioDataset', []);
      this.set('detailData', []);
      this.set('npuNodeName', '');
      this.set('benchNodeName', '');
      return;
    }
    const isNodeStartWithN = this._node.name.startsWith(NPU_PREFIX);
    const isSingleGraphNode = !isNodeStartWithN && !this._node?.name?.startsWith(BENCH_PREFIX);
    this.set('isSingleGraphNode', isSingleGraphNode);
    if (!isSingleGraphNode) {
      await this._updateMatchedNodeLink(this.selectedNode);
    }
    // 考虑选中的节点是匹配节点的情况
    const npuNode = isNodeStartWithN || isSingleGraphNode ? this._node : this._matchedNode;
    let benchNode;
    if (!isSingleGraphNode) {
      benchNode = isNodeStartWithN ? this._matchedNode : this._node;
    } else {
      benchNode = undefined;
    }
    this.set('npuNodeName', npuNode?.name?.replace(NPU_PREFIX, ''));
    this.set('benchNodeName', benchNode?.name?.replace(BENCH_PREFIX, ''));
    const inputDataset = this.useNodeInfo.getIoDataSet(npuNode, benchNode, 'inputData');
    const outputDataSet = this.useNodeInfo.getIoDataSet(npuNode, benchNode, 'outputData');
    const ioDataset = [
      ...inputDataset.matchedIoDataset,
      ...outputDataSet.matchedIoDataset,
      ...inputDataset.unMatchedNpuIoDataset,
      ...outputDataSet.unMatchedNpuIoDataset,
      ...inputDataset.unMatchedBenchIoDataset,
      ...outputDataSet.unMatchedBenchIoDataset,
    ];
    this.set('ioDataset', ioDataset);
    const detailData = this.useNodeInfo.getDetailDataSet(npuNode, benchNode);
    this.set('detailData', detailData);
  }

  _getNode(selectedNode): GroupNode | OpNode | undefined | null {
    if (!selectedNode) {
      return null;
    }
    const isBench = selectedNode.startsWith(BENCH_PREFIX);
    return isBench ? this.graphHierarchy?.bench?.node(selectedNode) : this.graphHierarchy?.npu.node(selectedNode);
  }

  async _updateMatchedNodeLink(selectedNode: string): Promise<void> {
    const isBench = selectedNode.startsWith(BENCH_PREFIX);
    const [selectedHierarchy, matchedHierarchy] = isBench
      ? [this.graphHierarchy?.bench, this.graphHierarchy?.npu]
      : [this.graphHierarchy?.npu, this.graphHierarchy?.bench];
    const matchedNodes: Array<string> = (selectedHierarchy?.node(selectedNode) as Metanode | OpNode)?.matchedNodeLink;
    if (isEmpty(matchedNodes)) {
      this.set('_matchedNode', null);
      return;
    }
    const matchNodeName = matchedNodes[matchedNodes.length - 1];
    let matchNode = matchNodeName ? matchedHierarchy?.node(matchNodeName) : null;
    // 如果没有匹配节点，则通过接口获取匹配节点的信息
    if (isEmpty(matchNode)) {
      const nodeInfo = {
        nodeName: matchNodeName?.replace(new RegExp(`^(${NPU_PREFIX}|${BENCH_PREFIX})`), ''), // 去掉前缀
        nodeType: this._node?.name?.startsWith(NPU_PREFIX) ? 'Bench' : 'NPU',
      };
      const { success, data, error } = await this.useNodeInfo.getNodeInfo(nodeInfo, this.selection);
      if (success) {
        this.set('_matchedNode', data);
      } else {
        Notification.show(`获取匹配节点失败：${error}`, {
          position: 'middle',
          duration: 2000,
          theme: 'error',
        });
      }
    } else {
      this.set('_matchedNode', matchNode);
    }
  }

  // 点击单元格高亮
  handleGridCellClick(e: MouseEvent, syncGrid: HTMLElement): void {
    const target = e.composedPath()[0] as HTMLElement; // 获取点击的目标元素
    const slotValue = target.getAttribute('slot'); // 提取 slot 属性
    if (!slotValue || !slotValue.startsWith('vaadin-grid-cell-content-')) {
      return;
    }
    const cellIndex = parseInt(slotValue.split('-').pop() || '0', 10);
    // 前8个元素是表头不可选中，所以跳过
    if (cellIndex <= 8) {
      return;
    }
    const highlightedCells = this.shadowRoot?.querySelectorAll('.highlight-cell');
    highlightedCells?.forEach((cell) => cell.classList.remove('highlight-cell')); // 清除所有高亮样式
    target.classList.add('highlight-cell'); // 添加高亮样式到当前单元格
  }
}
