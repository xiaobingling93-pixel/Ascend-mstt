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

import '@vaadin/button';
import '@vaadin/details';
import '@vaadin/combo-box';

import { isEmpty } from 'lodash';
import { Notification } from '@vaadin/notification';
import { PolymerElement, html } from '@polymer/polymer';
import * as tf_graph_render from '../../../tf_graph_common/render';
import { customElement, property, observe } from '@polymer/decorators';
import { NPU_PREFIX, BENCH_PREFIX } from '../../../tf_graph_common/common';
import useMatched from './useMatched';
import type { UseMatchedType } from './useMatched';
import '@vaadin/progress-bar';
import '../tf_search_combox/index';
@customElement('tf-manual-match')
class Legend extends PolymerElement {
  // 定义模板
  static get template(): HTMLTemplateElement {
    return html`
      <style>
        .matched-button {
          display: flex;
          justify-content: end;
          margin-top: 10px;
        }
        .matched-button button {
          border: 1px solid #ccc;
          cursor: pointer;
        }
        .vaadin-details-selected {
          display: flex;
          padding-top: 0;
        }
        vaadin-combo-box::part(input-field) {
          height: 30px;
          border: 1px solid var(--paper-input-container-color, var(--secondary-text-color));
          background-color: white;
          font-size: 14px;
        }
        vaadin-combo-box::part(toggle-button) {
          font-size: 14px;
        }

        tf-search-combox.matched-node::part(arraw-button) {
          margin-top: 28px !important;
        }

        .vaadin-details-title {
          font-size: 14px;
          color: #333333;
          font-weight: 600;
          margin-bottom: 0;
        }

        .vaadin-details vaadin-details-summary {
          font-size: 15px;
          color: #333333;
          font-weight: 600;
        }
        .match-button {
          display: flex;
          justify-content: center;
        }
        .match-button vaadin-button {
          width: 100%;
          font-weight: 600;
        }
        .warning {
          border: 1px dashed #000000;
          border-radius: 4px;
          padding: 4px;
          font-size: 14px;
        }
        .warning vaadin-button {
          width: 100%;
          font-weight: 600;
        }
        .warning span {
          color: red;
        }
      </style>
      <vaadin-details class="vaadin-details" summary="节点匹配" opened>
        <div class="warning">
          <p>注意：匹配结束后需要<span>点击保存按钮</span>，将操作后数据更新到文件中，<span>否则操作无效</span></p>
          <vaadin-button
            class="save-button"
            theme="primary warning small"
            on-click="_saveMatchedNodesLink"
            disabled="[[saveLoading]]"
            >保存</vaadin-button
          >
          <template is="dom-if" if="[[saveLoading]]">
            <vaadin-progress-bar indeterminate></vaadin-progress-bar>
          </template>
        </div>
        <div class="unmatched-node">
          <p class="vaadin-details-title">未匹配节点</p>
          <tf-search-combox
            label="调试侧([[npuUnMatchedNodes.length]])"
            items="[[npuUnMatchedNodes]]"
            selected-value="{{selectedNpuUnMatchedNode}}"
            on-select-change="[[_changeNpuUnMatchedNode]]"
            is-compare-graph="[[isCompareGraph]]"
          ></tf-search-combox>
          <tf-search-combox
            label="标杆侧([[benchUnMatchedNodes.length]])"
            items="[[benchUnMatchedNodes]]"
            selected-value="{{selectedBenchUnMatchedNode}}"
            on-select-change="[[_changeBenchUnMatchedNode]]"
            is-compare-graph="[[isCompareGraph]]"
          ></tf-search-combox>
          <div class="match-button">
            <vaadin-button theme="secondary small" on-click="_addMatchedNodesLink">点击匹配</vaadin-button>
          </div>
        </div>
        <div class="matched-node">
          <p class="vaadin-details-title">已匹配节点</p>
          <tf-search-combox
            class="matched-node"
            label="调试侧([[npuMatchedNodes.length]])"
            items="[[npuMatchedNodes]]"
            selected-value="{{selectedNpuMatchedNode}}"
            on-select-change="[[_changeNpuMatchedNode]]"
          ></tf-search-combox>
          <tf-search-combox
            class="matched-node"
            label="标杆侧([[benchMatchedNodes.length]])"
            items="[[benchMatchedNodes]]"
            selected-value="{{selectedBenchMatchedNode}}"
            on-select-change="[[_changeBenchMatchedNode]]"
          ></tf-search-combox>
          <div class="match-button">
            <vaadin-button theme="secondary small" on-click="_deletelMatchedNodesLink">取消匹配</vaadin-button>
          </div>
        </div>
      </vaadin-details>
    `;
  }

  @property({ type: Object })
  unmatched: any = [];

  @property({ type: Object })
  selection: any;

  @property({ type: Object })
  renderHierarchy: tf_graph_render.MergedRenderGraphInfo = {} as any;

  @property({ type: Boolean })
  isCompareGraph: boolean = true;

  @property({ type: String, notify: true })
  selectedNode: string = '';

  @property({ type: Array })
  npuMatchedNodes: Array<string> = [];

  @property({ type: Array })
  benchMatchedNodes: Array<string> = [];

  @property({ type: Array })
  npuUnMatchedNodes: Array<string> = [];

  @property({ type: Array })
  benchUnMatchedNodes: Array<string> = [];

  @property({ type: String })
  selectedNpuMatchedNode: string = '';

  @property({ type: String })
  selectedBenchMatchedNode: string = '';

  @property({ type: String })
  selectedNpuUnMatchedNode: string = '';

  @property({ type: String })
  selectedBenchUnMatchedNode: string = '';

  @property({ type: Boolean })
  saveLoading: boolean = false;

  useMatched: UseMatchedType = useMatched();

  npuMatchedNodeList = {};
  benchMatchedNodeList = {};

  @observe('unmatched')
  _observeUnmatchedNode(): void {
    this.set('npuUnMatchedNodes', this.unmatched[0]);
    this.set('benchUnMatchedNodes', this.unmatched[1]);
    this.set('selectedNpuUnMatchedNode', '');
    this.set('selectedBenchUnMatchedNode', '');
  }

  @observe('renderHierarchy')
  _observeRenderHierarchy(): void {
    const isCompareGraphTemp = this.renderHierarchy.bench?.renderedOpNames.some((name: string) =>
      name.startsWith(BENCH_PREFIX),
    );
    this.set('isCompareGraph', isCompareGraphTemp);
  }

  @observe('selection')
  async _observeSelection(): Promise<void> {
    if (isEmpty(this.selection)) {
      return;
    }
    const result = await this.useMatched.queryMatchedStateList(this.selection);
    if (result.success) {
      // 初始化已匹配节点列表
      const { npu_match_nodes, bench_match_nodes } = result.data;
      this.npuMatchedNodeList = npu_match_nodes;
      this.benchMatchedNodeList = bench_match_nodes;
      this.set('npuMatchedNodes', Object.keys(npu_match_nodes));
      this.set('benchMatchedNodes', Object.keys(bench_match_nodes));
    } else {
      Notification.show(`错误：${result.error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }

  @observe('selectedNode')
  _observeSelectedNode(): void {
    if (isEmpty(this.selectedNode)) {
      return;
    }
    if (this.selectedNode.startsWith(NPU_PREFIX)) {
      this.set('selectedNpuUnMatchedNode', this.selectedNode.replace(NPU_PREFIX, ''));
    } else if (this.selectedNode.startsWith(BENCH_PREFIX)) {
      this.set('selectedBenchUnMatchedNode', this.selectedNode.replace(BENCH_PREFIX, ''));
    }
  }

  // 一定要写箭头函数，不然父子组件传值this指向有问题
  _changeNpuUnMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = NPU_PREFIX + this.selectedNpuUnMatchedNode;
      this.set('selectedNode', node);
      // 展开对应侧节点
      this.set('selectedNode', BENCH_PREFIX + this.selectedBenchMatchedNode);
    } else {
      Notification.show('提示：单图节点不支持匹配', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  _changeBenchUnMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = BENCH_PREFIX + this.selectedBenchUnMatchedNode;
      this.set('selectedNode', node);
      // 展开对应侧节点
      this.set('selectedNode', NPU_PREFIX + this.selectedNpuMatchedNode);
    } else {
      Notification.show('提示：单图节点不支持匹配', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  _changeNpuMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = NPU_PREFIX + this.selectedNpuMatchedNode;
      this.set('selectedBenchMatchedNode', this.npuMatchedNodeList[this.selectedNpuMatchedNode]);
      this.set('selectedNode', node);
    } else {
      Notification.show('提示：单图节点不支持匹配', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  _changeBenchMatchedNode = (): void => {
    if (this.isCompareGraph) {
      const node = BENCH_PREFIX + this.selectedBenchMatchedNode;
      this.set('selectedNpuMatchedNode', this.benchMatchedNodeList[this.selectedBenchMatchedNode]);
      this.set('selectedNode', node);
    } else {
      Notification.show('提示：单图节点不支持匹配', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
    }
  };

  // 取消匹配
  async _deletelMatchedNodesLink(): Promise<void> {
    const result = await this.useMatched.deleteMatchedNodesLink(
      this.selectedNpuMatchedNode,
      this.selectedBenchMatchedNode,
      this.selection,
      this.renderHierarchy,
    );
    if (result.success) {
      // 更新匹配关系
      delete this.npuMatchedNodeList[this.selectedNpuMatchedNode];
      delete this.benchMatchedNodeList[this.selectedBenchMatchedNode];
      // 未匹配列表添加取消匹配的节点
      this.set('npuUnMatchedNodes', [...this.npuUnMatchedNodes, this.selectedNpuMatchedNode]);
      this.set('benchUnMatchedNodes', [...this.benchUnMatchedNodes, this.selectedBenchMatchedNode]);
      // 已匹配列表删除匹配成功的节点
      this.set('npuMatchedNodes', Object.keys(this.npuMatchedNodeList));
      this.set('benchMatchedNodes', Object.keys(this.benchMatchedNodeList));
      // 未匹配列表选择取消匹配的节点
      this.set('selectedNpuUnMatchedNode', this.selectedNpuMatchedNode);
      this.set('selectedBenchUnMatchedNode', this.selectedBenchMatchedNode);
      // 选中节点
      this.set('selectedNode', '');
      this.set('selectedNode', NPU_PREFIX + this.selectedNpuMatchedNode);
      // 已匹配列表清空选中的节点
      this.set('selectedNpuMatchedNode', '');
      this.set('selectedBenchMatchedNode', '');
      Notification.show('取消成功：对应节点状态已更新', {
        position: 'middle',
        duration: 3000,
        theme: 'success',
      });
    } else {
      Notification.show(`匹配失败：${result.error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }

  // 匹配节点
  async _addMatchedNodesLink(): Promise<void> {
    const result = await this.useMatched.addMatchedNodesLink(
      this.selectedNpuUnMatchedNode,
      this.selectedBenchUnMatchedNode,
      this.selection,
      this.renderHierarchy,
    );
    if (result.success) {
      // 更新匹配关系
      this.npuMatchedNodeList[this.selectedNpuUnMatchedNode] = this.selectedBenchUnMatchedNode;
      this.benchMatchedNodeList[this.selectedBenchUnMatchedNode] = this.selectedNpuUnMatchedNode;
      // 未匹配列表删除匹配成功的节点
      this.set(
        'npuUnMatchedNodes',
        this.npuUnMatchedNodes.filter((node) => node !== this.selectedNpuUnMatchedNode),
      );
      this.set(
        'benchUnMatchedNodes',
        this.benchUnMatchedNodes.filter((node) => node !== this.selectedBenchUnMatchedNode),
      );
      // 已匹配列表添加匹配成功的节点
      this.set('npuMatchedNodes', Object.keys(this.npuMatchedNodeList));
      this.set('benchMatchedNodes', Object.keys(this.benchMatchedNodeList));
      // 已匹配列表选择匹配成功的节点
      this.set('selectedNpuMatchedNode', this.selectedNpuUnMatchedNode);
      this.set('selectedBenchMatchedNode', this.selectedBenchUnMatchedNode);
      // 选中节点
      this.set('selectedNode', '');
      this.set('selectedNode', NPU_PREFIX + this.selectedNpuUnMatchedNode);
      // 未匹配列表清空选中的节点
      this.set('selectedNpuUnMatchedNode', '');
      this.set('selectedBenchUnMatchedNode', '');

      Notification.show('匹配成功：对应节点状态已更新', {
        position: 'middle',
        duration: 3000,
        theme: 'success',
      });
    } else {
      Notification.show(`匹配失败：${result.error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }

  // 保存
  async _saveMatchedNodesLink(): Promise<void> {
    this.set('saveLoading', true);
    const result = await this.useMatched.saveMatchedNodesLink(this.selection);
    this.set('saveLoading', false);
    if (result.success) {
      Notification.show('保存成功：文件已变更', {
        position: 'middle',
        duration: 3000,
        theme: 'success',
      });
    } else {
      Notification.show(`保存失败：${result.error}`, {
        position: 'middle',
        duration: 3000,
        theme: 'error',
      });
    }
  }
}
