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
==============================================================================*/
import { customElement, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import '../polymer/irons_and_papers';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import '../tf_graph/tf-graph';
import * as tf_graph from '../tf_graph_common/graph';
import * as tf_graph_hierarchy from '../tf_graph_common/hierarchy';
import * as tf_graph_render from '../tf_graph_common/render';
import '../tf_graph_node_info/index';
import * as _ from 'lodash';
import type { MinimapVis } from '../tf_graph_controls/tf-graph-controls';

/**
 * Element for putting tf-graph and tf-graph-info side by side.
 *
 * Example
 * <tf-graph-board graph=[[graph]]></tf-graph-board>
 */
@customElement('tf-graph-board')
class TfGraphBoard extends LegacyElementMixin(PolymerElement) {
  static readonly template = html`
    <style>
      ::host {
        display: block;
      }

      /deep/ .close {
        position: absolute;
        cursor: pointer;
        left: 15px;
        bottom: 15px;
      }

      .container {
        width: 100%;
        height: 100%;
        opacity: 1;
        display: flex;
        flex-direction: column;
      }

      .container.loading {
        cursor: progress;
        opacity: 0.1;
      }

      .container.loading.error {
        cursor: auto;
      }

      #info {
        position: absolute;
        right: 5px;
        top: 5px;
        padding: 0px;
        min-width: 400px;
        background-color: rgba(255, 255, 255, 0.9);
        @apply --shadow-elevation-2dp;
      }

      #graph-info {
        position: relative;
        right: 0px;
        top: 0px;
        max-width: 1000px;
        max-height: 90vh;
        min-width: 400px;
        padding: 0px;
      }

      .resize-handle {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 20px;
        height: 20px;
        cursor: ne-resize;
        z-index: 1;
      }

      #main {
        width: 100%;
        height: 100%;
      }

      #tab-info {
        background-color: #ffffff;
        border-top: 2px solidrgb(153, 152, 152);
      }

      #progress-bar {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
        position: absolute;
        top: 40px;
        left: 0;
        font-size: 13px;
      }

      #progress-msg {
        margin-bottom: 5px;
        white-space: pre-wrap;
        width: 400px;
      }

      paper-progress {
        width: 400px;
        --paper-progress-height: 6px;
        --paper-progress-active-color: #f3913e;
      }

      .context-menu {
        position: absolute;
        display: none;
        background-color: #e2e2e2;
        border-radius: 2px;
        font-size: 14px;
        min-width: 150px;
        border: 1px solid #d4d4d4;
      }

      /deep/ .context-menu ul {
        list-style-type: none;
        margin: 0;
        padding: 0;
        cursor: default;
      }

      /deep/ .context-menu ul li {
        padding: 4px 16px;
      }

      /deep/ .context-menu ul li:hover {
        background-color: #f3913e;
        color: white;
      }
    </style>
    <template is="dom-if" if="[[_isNotComplete(progress)]]">
      <div id="progress-bar">
        <div id="progress-msg">[[progress.msg]]</div>
        <paper-progress value="[[progress.value]]"></paper-progress>
      </div>
    </template>
    <div class$="[[_getContainerClass(progress)]]">
      <div id="main">
        <tf-graph
          id="graph"
          selection="[[selection]]"
          graph-hierarchy="{{graphHierarchy}}"
          hierarchy-params="[[hierarchyParams]]"
          render-hierarchy="{{renderHierarchy}}"
          selected-node="{{selectedNode}}"
          highlighted-node="{{_highlightedNode}}"
          progress="{{progress}}"
          handle-node-selected="[[handleNodeSelected]]"
          menu="[[menu]]"
          colorset="[[colorset]]"
          minimap-vis="[[minimapVis]]"
        ></tf-graph>
      </div>
      <div id="tab-info">
        <tf-graph-vaadin-tab
          class="tf-graph-node-info"
          graph-hierarchy="[[graphHierarchy]]"
          selected-node="{{selectedNode}}"
          tooltips="[[tooltips]]"
          selection="[[selection]]"
        >
        </tf-graph-vaadin-tab>
      </div>
    </div>
  `;

  @property({ type: Object })
  graphHierarchy: tf_graph_hierarchy.MergedHierarchy;

  @property({ type: Object })
  graph: tf_graph.MergedSlimGraph;

  @property({ type: Object })
  hierarchyParams: tf_graph_hierarchy.HierarchyParams = tf_graph_hierarchy.defaultHierarchyParams;

  /**
   * A number between 0 and 100 denoting the % of progress
   * for the progress bar and the displayed message.
   * @type {{value: number, msg: string}}
   */
  @property({ type: Object })
  progress: object;

  @property({ type: Object, notify: true })
  renderHierarchy: tf_graph_render.MergedRenderGraphInfo;

  @property({ type: Object })
  menu: any;

  @property({ type: Object })
  colorset: any;

  @property({ type: String, notify: true })
  selectedNode: string;

  @property({ type: String })
  _highlightedNode: string;

  // An optional function that takes a node selected event (whose `detail`
  // property is the selected node ... which could be null if a node is
  // deselected). Called whenever a node is selected or deselected.
  @property({ type: Object })
  handleNodeSelected: object;

  @property({ type: Object })
  selection: object;

  @property({ type: Object })
  tooltips: object;

  @property({ type: String })
  selectNodeCopy: string = '';

  @property({ type: Object })
  minimapVis: MinimapVis = { npu: true, bench: true };

  ready(): void {
    super.ready();
  }

  fit(): void {
    (this.$.graph as any).fit();
  }

  /** True if the progress is not complete yet (< 100 %). */
  _isNotComplete(progress): boolean {
    return progress.value < 100;
  }

  _getContainerClass(progress): string {
    let result = 'container';
    if (progress.error) {
      result += ' error';
    }
    if (this._isNotComplete(progress)) {
      result += ' loading';
    }
    return result;
  }
}
