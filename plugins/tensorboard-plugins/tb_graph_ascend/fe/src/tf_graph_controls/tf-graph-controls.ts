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
import '@vaadin/details';
import '@vaadin/select';
import './components/ts_linkage_search_combox/index';
import { customElement, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import * as _ from 'lodash';
import { DarkModeMixin } from '../polymer/dark_mode_mixin';
import '../polymer/irons_and_papers';
import './components/tf_manual_match/index';
import './components/tf_color_select/index';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import '../tf_dashboard_common/tensorboard-color';
import { SelectionType } from '../tf_graph_common/common';
import * as tf_graph_proto from '../tf_graph_common/proto';
import * as tf_graph_render from '../tf_graph_common/render';
import '../tf_graph_common/tf-graph-icon';
import '../tf_graph_loader/tf-graph-dashboard-loader';

export interface Selection {
  run: string;
  tag: string | null;
  type: SelectionType;
  batch: number;
  step: number;
}
export interface TagItem {
  tag: string | null;
  displayName: string;
  conceptualGraph: boolean;
  opGraph: boolean;
  profile: boolean;
}
export interface RunItem {
  name: string;
  tags: TagItem[];
}
export type Dataset = Array<RunItem>;
@customElement('tf-graph-controls')
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

      :host(.dark-mode) {
        --tb-graph-controls-title-color: #fff;
        --tb-graph-controls-legend-text-color: #f3f3f3;
        --tb-graph-controls-text-color: #eee;
      }

      paper-dropdown-menu {
        --paper-dropdown-menu-input: {
          padding: 0;
          color: gray;
        }
        --iron-icon-width: 15px;
        --iron-icon-height: 15px;
        --primary-text-color: gray;
        --paper-item-min-height: 30px;
      }

      vaadin-combo-box {
        flex:1;
        font-size: small;
      }

      vaadin-combo-box::part(input-field) {
        background-color: white;
        border: 1px solid #0d0d0d;
        height: 30px;
        border-radius: 0;
      }

      vaadin-select {
        width: 100px;
        padding-left: 15px;
      }

      vaadin-select::part(input-field) {
        background-color: white;
        border: 1px solid #0d0d0d;
        height: 30px;
      }

      vaadin-select-item {
        font-size: smaller;
      }

      paper-button[raised].keyboard-focus {
        font-weight: normal;
      }

      .run-dropdown {
        --paper-input-container: {
          padding: 5px 0 5px 5px;
        }
      }

      table {
        border-collapse: collapse;
        border-spacing: 0;
      }

      table tr {
        height: 20px;
      }

      table td {
        padding: 0;
        margin: 10px;
      }

      .allcontrols {
        flex-grow: 1;
        overflow-y: auto;
      }

      .container-search{
        display: flex;
        align-items: center;
        padding-right: 8px;
      }
      .holder {
        background: rgb(246,246,246);
        box-sizing: border-box;
        color: var(--tb-graph-controls-text-color);
        width: 100%;
      }

      paper-radio-button {
        display: block;
        padding: 5px;
      }

      .rectangle {
        width: 40px;
        height: 18px;
        display: inline-block;
        border-radius: 1px;
        margin-right: 10px;
      }

      .-rectangle {
        width: 35px;
        height: 16px;
        background-color: red;
        display: inline-block;
      }

      svg.icon,
      tf-graph-icon {
        width: 50px;
        height: 18px;
      }

      .image-icon {
        width: 24px;
        height: 24px;
      }

      .help-icon {
        height: 15px;
        margin: 0;
        padding: 0;
      }

      .gray {
        color: #666;
      }

      .title {
        font-size: var(--tb-graph-controls-title-font-size);
        margin: 8px 5px 0px 0;
        color: var(--tb-graph-controls-title-color);
      }
      .title small {
        font-weight: normal;
      }
      .container-wrapper{
        margin: 20px 0;
        border-top: 1px #bfbfbf dashed;
      }
      #file {
        padding: 8px 0;
      }

      .color-legend-row {
        align-items: center;
        clear: both;
        display: flex;
        height: 20px;
        margin-top: 5px;
      }

      .color-legend-row .label,
      .color-legend-row svg,
      .color-legend-row tf-graph-icon {
        flex: 0 0 40px;
        margin-right: 10px;
      }

      .control-holder .icon-button {
        font-size: var(--tb-graph-controls-subtitle-font-size);
        margin: 0 -5px;
        padding: 5px;
        display: flex;
        justify-content: flex-start;
        color: var(--tb-graph-controls-text-color);
      }

      .button-text {
        padding-left: 20px;
        text-transform: none;
      }

      .upload-button {
        width: 165px;
        height: 25px;
        text-transform: none;
        margin-top: 4px;
      }

      .button-icon {
        width: 26px;
        height: 26px;
        color: var(--paper-orange-500);
      }

      .hidden-input {
        display: none;
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

      .allcontrols .control-holder paper-toggle-button {
        margin-bottom: 5px;
      }

      span.counter {
        font-size: var(--tb-graph-controls-subtitle-font-size);
        color: gray;
        margin-left: 4px;
      }
      .counter-total {
        font-size: var(--tb-graph-controls-subtitle-font-size);
        color: gray;
        margin-left: 4px;
        width: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .runs-row .title,
      .tags-row .title {
        display: flex;
        align-items: baseline;
      }

      .runs-row paper-item,
      .tags-row paper-item {
        --paper-item: {
          white-space: nowrap;
        }
      }

      table.control-holder {
        border: 0;
        border-collapse: collapse;
      }

      table.tf-graph-controls td.input-element-table-data {
        padding: 0 0 0 20px;
      }

      .spacer {
        flex-grow: 1;
      }

      .color-text {
        overflow: hidden;
      }

      .color-text.gradient-container {
        margin: 0 5px;
      }

      /** Override inline styles that suppress pointer events for disabled buttons. Otherwise, the */
      /*  tooltips do not appear. */
      paper-radio-group paper-radio-button {
        pointer-events: auto !important;
      }

      paper-dropdown-menu {
        flex-grow: 1;
      }

      .tabs {
        display: flex;
        border-bottom: 1px solid #ccc;
      }

      .tab-button {
        flex: 1;
        padding: 10px;
        background:rgb(246, 246, 246);
        border: none;
        cursor: pointer;
        border-bottom: 3px solid transparent; /* 初始状态下的底部线条透明 */
      }

      .tab-button.active {
        /* background: #fff; */
        border-bottom: 3px solid orange; /* 聚焦时显示橙色线条 */
      }

      .matched-button{
        display:flex;
        justify-content: end;
        margin-top: 10px;
      }
      .matched-button button{
        border: 1px solid #ccc;
        cursor: pointer;
      }
      .tab-content {
        background:rgb(255,255,255);
        padding:0 20px;
        flex-grow: 1;
        overflow-y: auto;
      }

      .hidden {
        display: none;
      }

      .vaadin-details-matched vaadin-details-summary{
        font-size: 14px;
        color:black;
      }

      details {
        background: #fffcf3;
        margin: 0;
        padding: 0;
      }

      details > details {
        padding-left: 5px;
      }

      details > details > * {
        margin: 0;
      }

      details > details > details {
        padding-left: 10px; /* 每层嵌套增加 5px*/
      }

      summary {
        cursor: pointer;
        background: #fffcf3;
        white-space: nowrap; /* 防止内容换行 */
        overflow: hidden;
        text-overflow: ellipsis;
        padding:8px 4px;
        font-size: 14px;
      }

      .no-arrow::before {
        content: none;
      }

      summary::before {
        width: 0;
        height: 0;
        border-left: 4px solid black;
        border-right: 4px solid transparent;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        margin-right: 5px;
      }

      details[open] > summary::before {
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid black;
        border-bottom: 0;
        margin-right: 5px;
      }

      details > details > summary::before {
        border-left: 4px solid black;
        border-right: 4px solid transparent;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        margin-right: 5px;
      }

      details > details[open] > summary::before {
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid black;
        border-bottom: 0;
        margin-right: 5px;
      }

      details > details > details > summary::before {
        border-left: 4px solid black;
        border-right: 4px solid transparent;
        border-top: 4px solid transparent;
        border-bottom: 4px solid transparent;
        margin-right: 5px;
      }

      details > details > details[open] > summary::before {
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 4px solid black;
        border-bottom: 0;
        margin-right: 5px;
      }
      .vaadin-details vaadin-details-summary{
        font-size: 15px;
        color: #333333;
        font-weight: 600;
      }

      paper-checkbox {
        --paper-checkbox-unchecked-color: gray; /* 选中时的颜色 */
      }
    </style>

    <div class="tabs">
      <button class="tab-button" on-tap="_showNodeControls">设置</button>
      <button class="tab-button" on-tap="_showMatch">匹配</button>
    </div>
    <div id="nodes-content" class="tab-content">
      <div class="allcontrols">
        <div class="control-holder">
          <paper-button class="icon-button" on-tap="_fit" alt="Fit to screen">
            <iron-icon icon="aspect-ratio" class="button-icon"></iron-icon>
            <span class="button-text">Fit to screen</span>
          </paper-button>
        </div>
        <div class="control-holder">
          <paper-button class="icon-button" on-click="download" alt="Download PNG">
            <iron-icon icon="file-download" class="button-icon"></iron-icon>
            <span class="button-text">Download PNG</span>
          </paper-button>
        </div>
        <template is="dom-if" if="[[showUploadButton]]">
          <div class="control-holder">
            <paper-button
              class="icon-button"
              on-click="_getFile"
              alt="Upload file"
              title="Upload a pbtxt file to view a graph from the local filesystem"
            >
              <iron-icon icon="file-upload" class="button-icon"></iron-icon>
              <span class="button-text">Upload file</span>
            </paper-button>

            <div class="hidden-input">
              <input type="file" id="file" name="file" on-change="_updateFileInput" accept=".pbtxt" />
            </div>
          </div>
        </template>
        <div class="control-holder runs-row">
          <div class="title">Run <span class="counter">([[datasets.length]])</span></div>
          <paper-dropdown-menu no-label-float no-animations noink horizontal-align="left" class="run-dropdown">
            <paper-listbox class="dropdown-content" selected="{{_selectedRunIndex}}" slot="dropdown-content">
              <template is="dom-repeat" items="[[datasets]]">
                <paper-item>[[item.name]]</paper-item>
              </template>
            </paper-listbox>
          </paper-dropdown-menu>
        </div>
        <template is="dom-if" if="[[showSessionRunsDropdown]]">
          <div class="control-holder tags-row">
            <div class="title">
              Tag
              <span class="counter">([[_numTags(datasets, _selectedRunIndex)]])</span>
            </div>
            <paper-dropdown-menu no-label-float no-animations horizontal-align="left" noink class="run-dropdown">
              <paper-listbox class="dropdown-content" selected="{{_selectedTagIndex}}" slot="dropdown-content">
                <template is="dom-repeat" items="[[_getTags(datasets, _selectedRunIndex)]]">
                  <paper-item on-click="_clearMicroStep">[[item.displayName]]</paper-item>
                </template>
              </paper-listbox>
            </paper-dropdown-menu>
          </div>
        </template>
        <div class="control-holder runs-row">
          <template is="dom-if" if="{{steplist.length}}">
            <div class="title">Step <span class="counter">([[steplist.length]])</span></div>
            <paper-dropdown-menu no-label-float no-animations noink horizontal-align="left" class="run-dropdown">
              <paper-listbox class="dropdown-content" selected="{{_selectedStep}}" slot="dropdown-content">
                <template is="dom-repeat" items="[[steplist]]">
                  <paper-item>[[item]]</paper-item>
                </template>
              </paper-listbox>
            </paper-dropdown-menu>
          </template>
          <template is="dom-if" if="{{microsteps.length}}">
            <div class="title">MicroStep<span class="counter">([[computedLength(microsteps)]])</span></div>
            <paper-dropdown-menu no-label-float no-animations noink horizontal-align="left" class="run-dropdown">
              <paper-listbox class="dropdown-content" selected="{{_selectedMicroStep}}" slot="dropdown-content">
                <template is="dom-repeat" items="[[microsteps]]">
                  <paper-item>[[item]]</paper-item>
                </template>
              </paper-listbox>
            </paper-dropdown-menu>
          </template>
        </div>
      </div>
      <div class="container-wrapper">
        <tf-linkage-search-combox
          menu="[[menu]]"
          render-hierarchy="[[renderHierarchy]]"
          selected-node="{{selectedNode}}"
        >
        </tf-linkage-search-combox>
      </div>
      <tf-color-select
        menu="[[menu]]"
        unmatched="[[unmatched]]"
        render-hierarchy="[[renderHierarchy]]"
        colors="[[colors]]"
        overflowcheck="[[overflowcheck]]"
        selected-node="{{selectedNode}}"
        selection="[[selection]]"
        colorset="[[colorset]]"
        datasets="[[datasets]]"
      ></tf-color-select>
    </div>
    <div id="directory-content" class="tab-content hidden"></div>
    <div id="match-content" class="tab-content hidden">
      <tf-manual-match
        unmatched="[[unmatched]]"
        render-hierarchy="[[renderHierarchy]]"
        selected-node="{{selectedNode}}"
        selection="[[selection]]"
      ></tf-manual-match>
    </div>
  `;

  // 核心part
  @property({ type: Array, observer: '_datasetsChanged' })
  datasets: any = [];

  /**
   * @type {tf_graph_render.MergedRenderGraphInfo}
   */
  @property({ type: Object })
  renderHierarchy: tf_graph_render.MergedRenderGraphInfo;

  /**
   * @type {!Selection}
   */
  @property({
    type: Object,
    notify: true,
    readOnly: true,
    computed:
      '_computeSelection(datasets, _selectedRunIndex, _selectedTagIndex, _selectedGraphType, _selectedMicroStep, _selectedStep)',
  })
  selection: object;

  /**
   * @type {SelectionType}
   */
  @property({ type: String })
  _selectedGraphType: string = SelectionType.OP_GRAPH;

  @property({ type: Object })
  graphDef: any;

  // 下载文件
  @property({ type: Object })
  _downloadFilename = 'graph.png';

  // Run 路径选择
  @property({ type: Number, observer: '_selectedRunIndexChanged' })
  _selectedRunIndex: number = 0;

  // Tag选择
  @property({ type: Number, observer: '_selectedTagIndexChanged' })
  _selectedTagIndex: number = 0;

  @property({ type: Boolean })
  showSessionRunsDropdown: boolean = true;

  // MicroStep 选择 和 Step选择
  @property({ type: Number })
  _selectedMicroStep: number = -1;
  _selectedStep: number = -1;

  @property({ type: Object })
  microsteps: any;
  steplist: any;

  // 目录，全量节点数据，支撑各种节点的搜索
  @property({ type: Object })
  menu: any;

  // 颜色数据
  @property({ type: Object })
  colors: any;

  // 颜色图例
  @property({ type: Object })
  colorset;

  // 溢出检测标志
  @property({ type: Boolean })
  overflowcheck;

  // 节点匹配，未匹配部分节点
  @property({ type: Object })
  unmatched: any = [];

  // 上传文件
  @property({ type: Object, notify: true })
  selectedFile: object;

  @property({ type: Boolean })
  showUploadButton: boolean = false;

  // trace input to off-state
  @property({ type: Boolean, notify: true })
  traceInputs: boolean = false;

  @property({ type: String, notify: true })
  selectedNode: string | null = null;

  override ready(): void {
    super.ready();
    this._showTabContent('设置', 'nodes-content');
    document.addEventListener('contextMenuTag-changed', this._getTagChanged.bind(this));
  }

  _getTagChanged(contextMenuTag): void {
    this.set('_selectedTagIndex', contextMenuTag.detail);
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

  _numTags(datasets: Dataset, _selectedRunIndex: number): number {
    return this._getTags(datasets, _selectedRunIndex).length;
  }

  _getTags(datasets: Dataset, _selectedRunIndex: number): TagItem[] {
    if (!datasets || !datasets[_selectedRunIndex]) {
      return [];
    }
    return datasets[_selectedRunIndex].tags;
  }

  _fit(): void {
    this.fire('fit-tap');
  }

  download(): void {
    this.fire('download-image-requested', this._downloadFilename);
  }

  _clearMicroStep(): void {
    // 也清除一下MicroStep和Step
    this.set('_selectedMicroStep', -1);
    this.set('_selectedStep', -1);
    this.set('selectedNode', null);
  }

  computedLength(microsteps): number {
    return microsteps.length > 0 ? microsteps.length - 1 : 0;
  }

  _updateFileInput(e: Event): void {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) {
      return;
    }
    // Strip off everything before the last "/" and strip off the file
    // extension in order to get the name of the PNG for the graph.
    let filePath = file.name;
    const dotIndex = filePath.lastIndexOf('.');
    if (dotIndex >= 0) {
      filePath = filePath.substring(0, dotIndex);
    }
    const lastSlashIndex = filePath.lastIndexOf('/');
    if (lastSlashIndex >= 0) {
      filePath = filePath.substring(lastSlashIndex + 1);
    }
    this._setDownloadFilename(filePath);
    this.set('selectedFile', e);
  }

  _datasetsChanged(newDatasets: Dataset, oldDatasets: Dataset): void {
    if (oldDatasets !== null) {
      // Select the first dataset by default.
      this._selectedRunIndex = 0;
    }
    this._setDownloadFilename(this.datasets[this._selectedRunIndex]?.name);
  }

  _computeSelection(
    datasets: Dataset,
    _selectedRunIndex: number,
    _selectedTagIndex: number,
    _selectedGraphType: SelectionType,
    _selectedMicroStep: number,
    _selectedStep: number,
  ): { run: string; tag: string | null; type: SelectionType; batch: number; step: number } | null {
    if (!datasets[_selectedRunIndex] || !datasets[_selectedRunIndex].tags[_selectedTagIndex]) {
      return null;
    }
    return {
      run: datasets[_selectedRunIndex].name,
      tag: datasets[_selectedRunIndex].tags[_selectedTagIndex].tag,
      type: _selectedGraphType,
      batch: _selectedMicroStep,
      step: _selectedStep,
    };
  }

  _selectedRunIndexChanged(runIndex: number): void {
    if (!this.datasets) {
      return;
    }
    this._selectedTagIndex = 0;
    this._selectedGraphType = this._getDefaultSelectionType();
    this.traceInputs = false; // Set trace input to off-state.
    this._setDownloadFilename(this.datasets[runIndex]?.name);
  }

  _selectedTagIndexChanged(): void {
    this._selectedGraphType = this._getDefaultSelectionType();
  }

  _getDefaultSelectionType(): SelectionType {
    const { datasets: newDatasets, _selectedRunIndex: run, _selectedTagIndex: tag } = this;
    const shouldSkip = !newDatasets || !newDatasets[run] || !(newDatasets[run] as any).tags[tag] || (newDatasets[run] as any).tags[tag].opGraph;
    if (shouldSkip) {
      return SelectionType.OP_GRAPH;
    }
    const datasetForRun = newDatasets[run] as any;
    if (datasetForRun.tags[tag].profile) {
      return SelectionType.PROFILE;
    }
    if (datasetForRun.tags[tag].conceptualGraph) {
      return SelectionType.CONCEPTUAL_GRAPH;
    }
    return SelectionType.OP_GRAPH;
  }

  _getFile(): void {
    (this.$$('#file') as HTMLElement).click();
  }

  _setDownloadFilename(name?: string): void {
    this._downloadFilename = `${name ?? 'graph'}.png`;
  }

  _statsNotNull(stats: tf_graph_proto.StepStats): boolean {
    return stats !== null;
  }
}
