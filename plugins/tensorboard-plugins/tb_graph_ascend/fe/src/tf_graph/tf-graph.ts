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

import { customElement, observe, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import * as _ from 'lodash';
import '../polymer/irons_and_papers';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import * as tb_debug from '../tb_debug';
import * as tf_graph from '../tf_graph_common/graph';
import * as tf_graph_hierarchy from '../tf_graph_common/hierarchy';
import * as tf_graph_render from '../tf_graph_common/render';
import * as tf_graph_util from '../tf_graph_common/util';
import * as tf_graph_layout from '../tf_graph_common/layout';
import './tf-graph-scene';
import './components/legend/index';
import { Selection } from '../tf_graph_controls/tf-graph-controls';
import { fetchPbTxt, parseGraphPbTxt } from '../tf_graph_common/parser';
import * as tf_hierarchy from '../tf_graph_common/hierarchy';
import * as tf_graph_parser from '../tf_graph_common/parser';

import { BENCH_PREFIX } from '../tf_graph_common/common';
import { safeJSONParse } from '../utils';

let _isRankJump = '';

export function setRankJump(value: string): void {
  _isRankJump = value;
}

@customElement('tf-graph')
class TfGraph extends LegacyElementMixin(PolymerElement) {
  static readonly template = html`
    <style>
      .container {
        width: 100%;
        height: 100%;
        background: white;
        display: flex;
      }

      .vertical {
        height: 100%;
        flex: 1;
        @apply --layout-vertical;
      }

      .auto {
        position: relative;
        @apply --layout-flex-auto;
        @apply --layout-vertical;
      }

      .bench {
        width: 50%;
        height: 100%;
        border-left: 2px dashed #cccccc;
      }

      h2 {
        text-align: center;
      }

      paper-button {
        text-transform: none;
      }
    </style>
    <scene-legend></scene-legend>
    <div class="container">
      <div class="vertical">
        <template is="dom-if" if="[[title]]">
          <h2>[[title]]</h2>
        </template>
        <tf-graph-scene
          id="scene"
          class="auto"
          render-hierarchy="[[renderHierarchy.npu]]"
          linked-hierarchy="[[renderHierarchy.bench]]"
          highlighted-node="[[_getVisible(highlightedNode)]]"
          selected-node="{{selectedNode}}"
          linked-node="{{linkedNode}}"
          progress="[[progress]]"
          node-context-menu-items="[[nodeContextMenuItems]]"
        ></tf-graph-scene>
      </div>
      <template is="dom-if" if="[[graphHierarchy.bench]]">
        <div class="bench">
          <tf-graph-scene
            id="bench"
            class="auto"
            render-hierarchy="[[renderHierarchy.bench]]"
            linked-hierarchy="[[renderHierarchy.npu]]"
            highlighted-node="[[_getVisible(highlightedNode)]]"
            selected-node="{{selectedNode}}"
            linked-node="{{linkedNode}}"
            progress="[[progress]]"
            node-context-menu-items="[[nodeContextMenuItems]]"
          ></tf-graph-scene>
        </div>
      </template>
    </div>
  `;

  @property({
    type: Object,
    notify: true,
    observer: '_graphChanged',
  })
  graphHierarchy: tf_graph_hierarchy.MergedHierarchy;

  @property({ type: Object })
  hierarchyParams: tf_graph_hierarchy.HierarchyParams;

  @property({ type: Object, notify: true })
  progress: object;

  @property({ type: String })
  override title: string;

  @property({ type: String, notify: true })
  selectedNode: string;

  @property({ type: String, notify: true })
  linkedNode: string;

  @property({ type: Object })
  _lastSelectedEdgeGroup: any;

  @property({ type: Object })
  _lastHighlightedEdgeGroup: any;

  @property({ type: String, notify: true })
  highlightedNode: string;

  @property({ type: Object, notify: true })
  highlightedEdge: tf_graph_render.EdgeData;

  @property({
    type: Object,
    readOnly: true,
    notify: true,
  })
  renderHierarchy: tf_graph_render.MergedRenderGraphInfo;

  @property({ type: Array })
  nodeContextMenuItems: unknown[];

  @property({ type: Number })
  _renderDepth: number = 1;

  @property({ type: Boolean })
  _allowGraphSelect: boolean = true;

  @property({ type: Object })
  handleNodeSelected: any = '';

  @property({ type: Object })
  selection: Selection;

  @observe('graphHierarchy', 'handleNodeSelected')
  _buildNewRenderHierarchy(): void {
    let graphHierarchy = this.graphHierarchy;
    if (!graphHierarchy) {
      return;
    }
    this._buildRenderHierarchy(graphHierarchy);
  }

  @observe('selectedNode')
  // Called when the selected node changes, ie there is a new selected node or
  // the current one is unselected.
  _selectedNodeChanged(): void {
    let selectedNode = this.selectedNode;
    if (this.handleNodeSelected) {
      // A higher-level component provided a callback. Run it.
      this.handleNodeSelected(selectedNode);
    }
    if (!selectedNode) {
      return;
    }
    const [selectedHierarchy, linkedHierarchy] = (
      selectedNode.startsWith(BENCH_PREFIX)
        ? [this.renderHierarchy.bench, this.renderHierarchy.npu]
        : [this.renderHierarchy.npu, this.renderHierarchy.bench]
    ) as tf_graph_render.RenderGraphInfo[];
    let node = selectedHierarchy.hierarchy.node(selectedNode);
    if (!node) {
      return;
    }
    const linkNodes = node.nodeAttributes._linked_node;
    if (Array.isArray(linkNodes)) {
      let tempNode = '';
      let lastRenderNode: tf_graph_render.RenderNodeInfo | undefined;
      let lastExpandStatus = false;
      for (let linkNode of linkNodes) {
        const renderLinkedNode = linkedHierarchy.getRenderNodeByName(linkNode);
        // Expand all ancestors of the linked node.
        if (renderLinkedNode) {
          lastRenderNode = renderLinkedNode;
          lastExpandStatus = renderLinkedNode.expanded;
          renderLinkedNode.expanded = true;
          tempNode = linkNode;
        } else {
          break;
        }
      }
      if (lastRenderNode) {
        lastRenderNode.expanded = lastExpandStatus;
      }
      this.set('linkedNode', tempNode);
    } else {
      this.set('linkedNode', '');
    }
  }

  @observe('selectedNode')
  async _menuSelectedNodeExpand(): Promise<void> {
    function shouldSkip(renderHierarchy: any, selectedNode: any): boolean {
      const hasRenderHierarchy = !!renderHierarchy;
      const isNodeRendered =
        renderHierarchy?.npu?.renderedOpNames?.includes(selectedNode) ||
        renderHierarchy?.bench?.renderedOpNames?.includes(selectedNode);
      const hasSelectedNode = !!selectedNode;

      return !hasRenderHierarchy || isNodeRendered || !hasSelectedNode;
    }
    if (shouldSkip(this.renderHierarchy, this.selectedNode)) {
      return;
    } else {
      const current = this.selectedNode;
      const tempHierarchy = (
        current.startsWith(BENCH_PREFIX) ? this.renderHierarchy.bench : this.renderHierarchy.npu
      ) as tf_graph_render.RenderGraphInfo;
      const params = new URLSearchParams();
      if (this.selection.run) {
        params.set('run', this.selection.run);
      }
      if (this.selection.tag) {
        params.set('tag', this.selection.tag);
      }
      params.set('node', this.selectedNode);
      const nodeMap = tempHierarchy.hierarchy.getNodeMap();
      const expandnodesPath = `expandnodes?${String(params)}`;

      let nodeName = '';
      try {
        const expandnodeStr = await tf_graph_parser.fetchPbTxt(expandnodesPath);
        let expandnodes;
        try {
          expandnodes = safeJSONParse(new TextDecoder().decode(expandnodeStr).replace(/'/g, '"')) as object;
        } catch (e) {
          console.error('Get expandnodes failed, please check expanded function and the nodedata in vis file');
        }
        if (expandnodes[1].length === 0 && expandnodes[2].length === 0) {
          return;
        }
        for (const i of expandnodes[1]) {
          nodeName = expandnodes[0] + i;
          const renderNode = tempHierarchy.getRenderNodeByName(nodeName);
          if (nodeName in nodeMap && !renderNode.expanded) {
            await this._nodeToggleExpand({ detail: { name: nodeName } });
          }
        }
        this.async(() => {
          try {
            this.set('selectedNode', ''); // 临时清空
            this.set('selectedNode', current); // 恢复原值
          } catch (e) {
            console.error('Error during async set operation:', e);
          }
        }, 175); // 代码会在延迟 175 毫秒后执行, 给浏览器足够的时间来处理多层展开带来的渲染和状态变化
      } catch (error) {
        console.error('Error fetching expandnodesPath:', error);
      }
    }
  }

  /**
   * Pans to a node. Assumes that the node exists.
   * @param nodeName {string} The name of the node to pan to.
   */
  panToNode(nodeName): void {
    (this.$$('tf-graph-scene') as any).panToNode(nodeName);
  }

  ready(): void {
    super.ready();

    this.addEventListener('graph-select', this._graphSelected.bind(this));
    this.addEventListener('disable-click', this._disableClick.bind(this));
    this.addEventListener('enable-click', this._enableClick.bind(this));
    // Nodes
    this.addEventListener('node-toggle-expand', this._nodeToggleExpand.bind(this));
    document.addEventListener('parent-node-toggle-expand', this._parentNodeToggleExpand.bind(this));
    this.addEventListener('node-select', this._nodeSelected.bind(this));
    this.addEventListener('node-highlight', this._nodeHighlighted.bind(this));
    this.addEventListener('node-unhighlight', this._nodeUnhighlighted.bind(this));
    this.addEventListener('node-toggle-extract', this._nodeToggleExtract.bind(this));

    // Annotations

    /* Note: currently highlighting/selecting annotation node has the same
     * behavior as highlighting/selecting actual node so we point to the same
     * set of event listeners.  However, we might redesign this to be a bit
     * different.
     */
    this.addEventListener('annotation-select', this._nodeSelected.bind(this));
    this.addEventListener('annotation-highlight', this._nodeHighlighted.bind(this));
    this.addEventListener('annotation-unhighlight', this._nodeUnhighlighted.bind(this));
  }

  _buildRenderHierarchy(graphHierarchy: tf_graph_hierarchy.MergedHierarchy): void {
    if (
      graphHierarchy.npu.root.type !== tf_graph.NodeType.META &&
      graphHierarchy.bench?.root.type !== tf_graph.NodeType.META
    ) {
      // root must be metanode but sometimes Polymer's dom-if has not
      // remove tf-graph element yet in <tf-node-info>
      // and thus mistakenly pass non-metanode to this module.
      return;
    }

    // Certain Polymer property setter are dynamically generated and is not properly
    // typed.
    const anyThis = this as any;

    const renderGraph = tf_graph_util.time(
      'new tf_graph_render.Hierarchy',
      () => {
        const npuRenderGraph = new tf_graph_render.RenderGraphInfo(graphHierarchy.npu);
        const mergedRenderGraph: tf_graph_render.MergedRenderGraphInfo = { npu: npuRenderGraph };
        if (graphHierarchy.bench) {
          const benchRenderGraph = new tf_graph_render.RenderGraphInfo(graphHierarchy.bench);
          mergedRenderGraph.bench = benchRenderGraph;
        }
        return mergedRenderGraph;
      },
      tb_debug.GraphDebugEventId.RENDER_BUILD_HIERARCHY,
    );
    setTimeout(() => {
      if (_isRankJump) {
        this.fire('node-select', { name: _isRankJump });
        _isRankJump = '';
      }
    }, tf_graph_layout.PARAMS.animation.duration);
    anyThis._setRenderHierarchy(renderGraph);
  }

  _getVisible(name: string): string {
    if (!name) {
      return name;
    }
    const tempHierarchy = (
      name.startsWith(BENCH_PREFIX) ? this.renderHierarchy.bench : this.renderHierarchy.npu
    ) as tf_graph_render.RenderGraphInfo;
    return tempHierarchy.getNearestVisibleAncestor(name);
  }

  fit(): void {
    (this.$.scene as any).fit();
    (this.$$('#bench') as any).fit();
  }

  getImageBlob(): Promise<Blob> {
    return (this.$.scene as any).getImageBlob();
  }

  _graphChanged(): void {
    if (!this.graphHierarchy) {
      return;
    }

    // When a new graph is loaded, fire this event so that there is no
    // info-card being displayed for the previously-loaded graph.
    this.fire('graph-select');
  }

  _graphSelected(event): void {
    // Reset this variable as a bug in d3 zoom behavior can cause zoomend
    // callback not to be called if a right-click happens during a zoom event.
    this._allowGraphSelect = true;
  }

  _disableClick(event): void {
    this._allowGraphSelect = false;
  }

  _enableClick(event): void {
    this._allowGraphSelect = true;
  }

  // Called only when a new (non-null) node is selected.
  _nodeSelected(event): void {
    this.set('selectedNode', event.detail.name);
  }

  _nodeHighlighted(event): void {
    this.set('highlightedNode', event.detail.name);
  }

  _nodeUnhighlighted(event): void {
    this.set('highlightedNode', null);
  }

  async _parentNodeToggleExpand(event): Promise<void> {
    const nodeName = event.detail.nodeData.node.name;
    const matchedNodeLink = event.detail.nodeData.node.matchedNodeLink;
    if (matchedNodeLink) {
      let matched = matchedNodeLink[matchedNodeLink.length - 1];
      this.set('selectedNode', matched);
    } else {
      const params = new URLSearchParams();
      params.set('run', this.selection.run);
      params.set('node', nodeName);
      if (this.selection.tag) {
        params.set('tag', this.selection.tag);
      }
      params.set('batch', String(this.selection.batch === -1 ? -1 : this.selection.batch - 1));
      params.set('step', String(this.selection.step === -1 ? -1 : this.selection.step - 1));
      const parentPath = `parent?${String(params)}`;
      const parentStr = await tf_graph_parser.fetchPbTxt(parentPath);
      const parentNode = new TextDecoder().decode(parentStr).replace(/'/g, '"');
      this.set('selectedNode', parentNode);
    }
  }

  async _nodeToggleExpand(event): Promise<void> {
    // Immediately select the node that is about to be expanded.
    // Compute the sub-hierarchy scene.
    const nodeName = event.detail.name;
    const isBench = nodeName.startsWith(BENCH_PREFIX);
    const tempHierarchy = (
      isBench ? this.renderHierarchy.bench : this.renderHierarchy.npu
    ) as tf_graph_render.RenderGraphInfo;
    const renderNode = tempHierarchy.getRenderNodeByName(nodeName);
    // Op nodes are not expandable.
    if (renderNode.node.type === tf_graph.NodeType.OP) {
      return;
    }
    if (!renderNode.expanded && !tempHierarchy.checkSubhierarchy(nodeName)) {
      const params = new URLSearchParams();
      params.set('run', this.selection.run);
      params.set('node', renderNode.node.name || '');
      if (this.selection.tag) {
        params.set('tag', this.selection.tag);
      }
      params.set('batch', String(this.selection.batch === -1 ? -1 : this.selection.batch - 1));
      params.set('step', String(this.selection.step === -1 ? -1 : this.selection.step - 1));
      const graphPath = `subgraph?${String(params)}`;
      const arrayBuffer = await fetchPbTxt(graphPath); // 等待 fetchPbTxt 完成
      const graphDef = await parseGraphPbTxt(arrayBuffer); // 等待 parseGraphPbTxt 完成
      const slimGraph = await tf_graph.build(graphDef, tf_graph.defaultBuildParams, undefined); // 等待 tf_graph.build 完成
      tf_hierarchy.update(tempHierarchy.hierarchy, slimGraph, nodeName);
      tempHierarchy.buildSubhierarchy(nodeName, slimGraph);
      renderNode.expanded = !renderNode.expanded;
      this.async(() => {
        (isBench ? this.$$('#bench') : (this.$.scene as any)).setNodeExpanded(renderNode);
      }, 75);
    } else {
      renderNode.expanded = !renderNode.expanded;
      this.async(() => {
        (isBench ? this.$$('#bench') : (this.$.scene as any)).setNodeExpanded(renderNode);
      }, 75);
    }
  }

  _nodeToggleExtract(event): void {
    // Toggle the include setting of the specified node appropriately.
    const nodeName = event.detail.name;
    this.nodeToggleExtract(nodeName);
  }

  nodeToggleExtract(nodeName: string): void {
    const tempHierarchy = (
      nodeName.startsWith(BENCH_PREFIX) ? this.renderHierarchy.bench : this.renderHierarchy.npu
    ) as tf_graph_render.RenderGraphInfo;
    const renderNode = tempHierarchy.getRenderNodeByName(nodeName);
    if (renderNode.node.include === tf_graph.InclusionType.INCLUDE) {
      renderNode.node.include = tf_graph.InclusionType.EXCLUDE;
    } else if (renderNode.node.include === tf_graph.InclusionType.EXCLUDE) {
      renderNode.node.include = tf_graph.InclusionType.INCLUDE;
    } else {
      renderNode.node.include = tempHierarchy.isNodeAuxiliary(renderNode)
        ? tf_graph.InclusionType.INCLUDE
        : tf_graph.InclusionType.EXCLUDE;
    }
    // Rebuild the render hierarchy.
    this._buildRenderHierarchy(this.graphHierarchy);
  }
}
