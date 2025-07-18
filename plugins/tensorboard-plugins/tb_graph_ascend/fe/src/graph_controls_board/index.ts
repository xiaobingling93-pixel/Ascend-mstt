/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (c) 2025, Huawei Technologies.
Adapt to the model hierarchical visualization data collected by the msprobe tool
==============================================================================*/

import '@vaadin/icon';
import '@vaadin/icons';
import '@vaadin/select';
import '@vaadin/button';

import * as _ from 'lodash';
import { customElement, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import { DarkModeMixin } from '../polymer/dark_mode_mixin';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import { PaperCheckboxElement } from '../polymer/irons_and_papers';
import '../polymer/irons_and_papers';

import './components/tf_main_controler';
import './components/tf_manual_match/index';
import './components/tf_color_select/index';
import './components/tf_linkage_search_combox/index';
import type { MetaDirType, MinimapVis } from './type';

@customElement('graph-controls-board')
class TfGraphControls extends LegacyElementMixin(DarkModeMixin(PolymerElement)) {
  static readonly template = html`
    <style>
      :host {
        color: #555;
        display: flex;
        flex-direction: column;
        font-size: 12px;
        width: 100%;
        --tb-graph-controls-title-color: #000;
        --tb-graph-controls-legend-text-color: #000;
        --tb-graph-controls-text-color: #555;
        --tb-graph-controls-title-font-size: 14px;
        --tb-graph-controls-subtitle-font-size: 14px;
        --paper-input-container-shared-input-style_-_font-size: 14px;
        --paper-font-subhead_-_font-size: 14px;
      }

      .holder {
        background: rgb(246, 246, 246);
        box-sizing: border-box;
        color: var(--tb-graph-controls-text-color);
        width: 100%;
      }

      .title {
        font-size: var(--tb-graph-controls-title-font-size);
        margin: 8px 5px 0px 0;
        color: var(--tb-graph-controls-title-color);
      }
      .title small {
        font-weight: normal;
      }
      .container-wrapper {
        margin: 10px 0 20px 0;
        border-top: 1px #bfbfbf dashed;
      }
      .minimap-control {
        font-size: var(--tb-graph-controls-title-font-size);
        height: 36px;
        line-height: 36px;
      }
      .right-checkbox {
        margin-left: 8px;
      }

      .icon-button {
        font-size: var(--tb-graph-controls-title-font-size);
      }
      .button-text {
        padding-left: 12px;
        text-transform: none;
      }
      .button-icon {
        width: 26px;
        height: 26px;
        margin-right: 5px;
        color: var(--paper-orange-500);
        font-weight: 500;
      }

      .allcontrols .control-holder {
        clear: both;
        display: flex;
        justify-content: space-between;
      }

      .allcontrols .control-holder.control-options {
        padding: 0 0 15px 15px;
        flex-direction: column;
      }

      .tabs {
        display: flex;
        border-bottom: 1px solid #ccc;
      }

      .tab-button {
        flex: 1;
        padding: 10px;
        background: rgb(246, 246, 246);
        border: none;
        cursor: pointer;
        border-bottom: 3px solid transparent; /* 初始状态下的底部线条透明 */
      }

      .tab-button.active {
        border-bottom: 3px solid orange; /* 聚焦时显示橙色线条 */
      }

      .tab-content {
        background: rgb(255, 255, 255);
        padding: 0 20px;
        flex-grow: 1;
        overflow-y: auto;
      }
      .fit-screen {
        display: flex;
        align-items: center;
      }
      .fit-screen vaadin-icon {
        margin-right: 10px;
        cursor: pointer;
        color: var(--paper-orange-500);
      }
      .fit-screen vaadin-button {
        margin-top: 6px;
        font-size: var(--tb-graph-controls-title-font-size);
        font-weight: 400;
        cursor: pointer;
      }

      .hidden {
        display: none;
      }

      .sync-expand-checkbox {
        font-size: var(--tb-graph-controls-title-font-size);
        margin-left: -4px;
      }
      paper-checkbox {
        --paper-checkbox-unchecked-color: gray; /* 选中时的颜色 */
        user-select: none;
      }
    </style>

    <div class="tabs">
      <button class="tab-button" on-tap="_showNodeControls">设置</button>
      <button class="tab-button" on-tap="_showMatch">匹配</button>
    </div>
    <div id="nodes-content" class="tab-content">
      <div class="fit-screen">
        <vaadin-icon icon="vaadin:viewport" on-click="_clickSetting"></vaadin-icon>
        <vaadin-button theme="tertiary contrast" on-click="_fit">自适应屏幕</vaadin-button>
      </div>

      <div class="minimap-control">
        <paper-checkbox checked on-change="_toggleNpuMinimap">调试侧缩略图</paper-checkbox>
        <template is="dom-if" if="[[!isSingleGraph]]">
          <paper-checkbox class="right-checkbox" checked on-click="_toggleBenchMinimap">标杆侧缩略图</paper-checkbox>
        </template>
      </div>
      <vaadin-checkbox class="sync-expand-checkbox" label="是否同步展开对应侧节点" checked={{isSyncExpand}}></vaadin-checkbox>
      <div class="container-wrapper">
        <tf-main-controler
          meta-dir="[[metaDir]]"
          selection="{{selection}}"
          microsteps="[[microsteps]]"
        ></tf-main-controler>
      </div>
      <div class="container-wrapper">
        <tf-linkage-search-combox
          nodelist="[[nodelist]]"
          selected-node="{{selectedNode}}"
          is-compare-graph="[[!isSingleGraph]]"
        >
        </tf-linkage-search-combox>
      </div>
      <tf-color-select
        colors="{{colors}}"
        task=[[task]]
        is-overflow-filter="{{isOverflowFilter}}"
        overflowcheck="[[overflowcheck]]"
        selected-node="{{selectedNode}}"
        selection="[[selection]]"
        colorset="[[colorset]]"
        is-single-graph="[[isSingleGraph]]"
      ></tf-color-select>
    </div>
    <div id="directory-content" class="tab-content hidden"></div>
    <div id="match-content" class="tab-content hidden">
      <tf-manual-match
        unmatched="[[unmatched]]"
        selected-node="{{selectedNode}}"
        selection="[[selection]]"
        is-compare-graph="[[!isSingleGraph]]"
        npu-match-nodes="[[npuMatchNodes]]"
        bench-match-nodes="[[benchMatchNodes]]"
        matched-config-files="[[matchedConfigFiles]]"
      ></tf-manual-match>
    </div>
  `;

  @property({ type: Object })
  metaDir: MetaDirType = {} as MetaDirType;

  @property({ type: Boolean, notify: true })
  isOverflowFilter: boolean = false;

  @property({ type: Boolean })
  isSingleGraph = false;

  @property({ type: Object, notify: true })
  selection: Selection = {} as Selection;

  // 全量节点数据，支撑各种节点的搜索
  @property({ type: Object })
  nodelist: object = {};

  // 颜色图例
  @property({ type: Object })
  colorset;

  // 溢出检测标志
  @property({ type: Boolean })
  overflowcheck;

  // 节点匹配，未匹配部分节点
  @property({ type: Object })
  unmatched: any = [];

  @property({ type: String, notify: true })
  selectedNode: string | null = null;

  @property({ type: Object, notify: true })
  minimapVis: MinimapVis = { npu: true, bench: true };

  @property({ type: Boolean, notify: true })
  isSyncExpand: boolean = true;

  // 颜色数据
  @property({ type: Object, notify: true })
  colors: any;

  override ready(): void {
    super.ready();
    this._showTabContent('设置', 'nodes-content');
  }

  _showTabContent(buttonText, contentId): void {
    // Remove 'active' class from all buttons
    this.shadowRoot?.querySelectorAll('.tab-button').forEach((button) => {
      button.classList.remove('active');
    });

    // Add 'active' class to the clicked button
    const buttons = this.shadowRoot?.querySelectorAll('.tab-button');
    buttons?.forEach((button) => {
      if ((button as HTMLElement).innerHTML === buttonText) {
        button?.classList.add('active');
      }
    });

    // Hide all content
    this.shadowRoot?.querySelectorAll('.tab-content').forEach((content) => {
      content.classList.add('hidden');
    });

    // Show the selected content
    const selectedContent = this.shadowRoot?.getElementById(contentId);
    if (selectedContent) {
      selectedContent.classList.remove('hidden');
    }
  }

  // 使用示例
  _showNodeControls(): void {
    this._showTabContent('设置', 'nodes-content');
  }

  _showMatch(): void {
    this._showTabContent('匹配', 'match-content');
  }

  _fit(): void {
    this.fire('fit-tap');
  }

  _toggleNpuMinimap(event: CustomEvent): void {
    const checkbox = event.target as PaperCheckboxElement;
    this.set('minimapVis.npu', checkbox.checked);
  }

  _toggleBenchMinimap(event: CustomEvent): void {
    const checkbox = event.target as PaperCheckboxElement;
    this.set('minimapVis.bench', checkbox.checked);
  }
}
