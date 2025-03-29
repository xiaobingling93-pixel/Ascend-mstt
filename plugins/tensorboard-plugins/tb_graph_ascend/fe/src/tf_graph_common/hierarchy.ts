/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright (c) 2025, Huawei Technologies.
Adapt to the model hierarchical visualization data collected by the msprobe tool
==============================================================================*/
/**
 * Package for the Graph Hierarchy for TensorFlow graph.
 */
import * as _ from 'lodash';
import * as tb_debug from '../tb_debug';
import { ProgressTracker } from './common';
import * as tf_graph from './graph';
import {
  createMetanode,
  GroupNode,
  Metanode,
  OpNode,
  ROOT_NAME,
  SlimGraph,
  MetanodeImpl,
} from './graph';
import * as tf_graph_util from './util';

export enum HierarchyEvent {
  /**
   * Fired when the templates may have been updated. No event payload attached.
   */
  TEMPLATES_UPDATED = 0,
}

// Including both the NPU and the benchmark hierarchy.
export interface MergedHierarchy {
  npu: Hierarchy;
  bench?: Hierarchy;
}
/**
 * Class for the Graph Hierarchy for TensorFlow graph.
 */
export class Hierarchy extends tf_graph_util.Dispatcher<HierarchyEvent> {
  root: Metanode;
  /**
   * Options passed to dagre for creating the graph. Note that the
   * `compound` argument will be overridden to true.
   */
  graphOptions: tf_graph.LabeledGraphOptions = {};

  private index: {
    [nodeName: string]: GroupNode | OpNode;
  };

  constructor(params: HierarchyParams) {
    super();
    this.graphOptions.compound = true;
    this.graphOptions.rankdir = params.rankDirection;
    this.root = createMetanode(ROOT_NAME, this.graphOptions);
    this.index = {};
    this.index[ROOT_NAME] = this.root;
  }

  getNodeMap(): {
    [nodeName: string]: GroupNode | OpNode;
  } {
    return this.index;
  }

  node(name: string): GroupNode | OpNode {
    return this.index[name];
  }

  setNode(name: string, node: GroupNode | OpNode): void {
    this.index[name] = node;
  }
}

export interface HierarchyParams {
  verifyTemplate: boolean;
  rankDirection: string;
}

export const defaultHierarchyParams: HierarchyParams = {
  verifyTemplate: true,
  rankDirection: 'TB',
};

/**
 * @param graph The raw graph.
 * @param params Parameters used when building a hierarchy.
 */
export function build(
  graph: tf_graph.SlimGraph,
  params: HierarchyParams,
  tracker?: ProgressTracker,
): Promise<Hierarchy> {
  const h = new Hierarchy(params);
  return tf_graph_util
    .runAsyncTask(
      'Adding nodes',
      100,
      () => {
        addNodesInVis(h, graph, ROOT_NAME);
      },
      tracker,
      tb_debug.GraphDebugEventId.HIERARCHY_ADD_NODES,
    )
    .then(() => {
      return h;
    });
}

/**
 * Updates hierarchy when the subgraph of a node is built.
 * @param oldGraph
 * @param slimGraph
 */
export function update(oldGraph: Hierarchy, slimGraph: tf_graph.SlimGraph, nodeName: string): void {
  let node = oldGraph.node(nodeName) as Metanode;
  if (node) {
    addNodesInVis(oldGraph, slimGraph, nodeName);
  }
}

/**
 * Creates the metanodes in the hierarchical graph and assigns parent-child
 * relationship between them in vis mode.
 * @param h
 * @param graph
 * @param parentName
 */
function addNodesInVis(h: Hierarchy, graph: SlimGraph, parentName: string): void {
  const parentNode = h.node(parentName);
  if (!(parentNode instanceof MetanodeImpl)) {
    return;
  }
  const orderedNodes: Array<{ idx: number; name: string; node: any }> = [];
  _.each([graph.nodes, graph.metaNodes], (nodes) => {
    _.each(nodes, (node) => {
      node.parentNode = parentNode;
      orderedNodes.push({
        idx: node.nodeAttributes._order ?? 0,
        name: node.name,
        node,
      });
      h.setNode(node.name, node);
    });
  });
  _.each(
    orderedNodes.sort((a, b) => a.idx - b.idx),
    (item) => {
      parentNode.metagraph.setNode(item.name, item.node);
    },
  );
}
