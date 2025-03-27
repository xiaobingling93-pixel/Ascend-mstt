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
import { graphlib } from 'dagre';
import * as _ from 'lodash';
import * as tb_debug from '../tb_debug';
import { ProgressTracker } from './common';
import * as tf_graph from './graph';
import {
  createGraph,
  createMetaedge,
  createMetanode,
  Edges,
  GraphType,
  GroupNode,
  Metaedge,
  MetaedgeImpl,
  Metanode,
  Node,
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
   * Whether at least one tensor in the graph has shape information.
   */
  hasShapeInfo = false;
  /**
   * The maximum size across all meta edges. Used for scaling thickness.
   */
  maxMetaEdgeSize = 1;
  orderings: {
    [nodeName: string]: {
      [childName: string]: number;
    };
  };
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
    this.orderings = {};
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

  /** Given the name of a node, return its incoming metaedges. */
  getPredecessors(nodeName: string): Edges {
    let node = this.index[nodeName];
    if (!node) {
      throw Error(`Could not find node with name: ${nodeName}`);
    }
    let predecessors = this.getOneWayEdges(node, true);
    // Add embedded predecessors, such as constants.
    if (!node.isGroupNode) {
      _.each((<OpNode>node).inEmbeddings, (embeddedNode) => {
        _.each((<OpNode>node).inputs, (input) => {
          if (input.name === embeddedNode.name) {
            // Make a new metaedge holding the edge between the
            // node and the in-embedding.
            let metaedge = new MetaedgeImpl(embeddedNode.name, nodeName);
            metaedge.addBaseEdge(
              {
                outputTensorKey: input.outputTensorKey,
                isReferenceEdge: false,
                v: embeddedNode.name,
                w: nodeName,
              },
              this,
            );
            predecessors.regular.push(metaedge);
          }
        });
      });
    }
    return predecessors;
  }

  /**
   * Given the name of a node, return its outgoing metaedges.
   *
   * This is the inverse of getPredecessors(). See that method's documentation
   * for an in-depth example.
   */
  getSuccessors(nodeName: string): Edges {
    let node = this.index[nodeName];
    if (!node) {
      throw Error(`Could not find node with name: ${nodeName}`);
    }
    let successors = this.getOneWayEdges(node, false);
    // Add embedded successors, such as summaries.
    if (!node.isGroupNode) {
      _.each((<OpNode>node).outEmbeddings, (embeddedNode) => {
        _.each(embeddedNode.inputs, (input) => {
          if (input.name === nodeName) {
            // Make a new metaedge holding the edge between the
            // node and the out-embedding.
            let metaedge = new MetaedgeImpl(nodeName, embeddedNode.name);
            metaedge.addBaseEdge(
              {
                outputTensorKey: input.outputTensorKey,
                isReferenceEdge: false,
                v: nodeName,
                w: embeddedNode.name,
              },
              this,
            );
            successors.regular.push(metaedge);
          }
        });
      });
    }
    return successors;
  }

  /**
   * For a given GroupNode, get or calculate an object which describes a
   * topological ordering of child nodes within that GroupNode's metagraph.
   *
   * This ordering is used when rendering bridge control edges which are
   * sometimes backwards relative to the dataflow.
   *
   * For example, say we have a graph with two edges A->B and A->C, and we're
   * interested in the ordering under ROOT. In this case, any of the following
   * would be legitimate return values:
   *
   *  - { 'A': 0, 'B': 1, 'C': 2 } -- most likely
   *  - { 'A': 0, 'B': 2, 'C': 1 } -- less likely
   *  - { 'A': 12, 'B': 100, 'C': 99 } -- unlikely, but still OK
   *
   * The algorithm does not guarantee that all numbers from 0-N (where N is
   * the number of nodes) appear exactly once. Rather it guarantees that if
   * there is a path between two nodes, the earlier one will have a lower
   * number in the ordering hash.
   *
   * When generating the ordering, we ignore control Metaedges (those which
   * represent only BaseEdges that have isControlDependency set to true).
   *
   * If there is no node with the specified name, an error is thrown. If the
   * node with the specified name is not a group node, null is returned.
   */
  getTopologicalOrdering(nodeName: string): {
    [childName: string]: number;
  } {
    let node = this.index[nodeName];
    if (!node) {
      throw Error(`Could not find node with name: ${nodeName}`);
    }
    if (!node.isGroupNode) {
      return {};
    }
    if (nodeName in this.orderings) {
      return this.orderings[nodeName];
    }
    // Mapping of a child node names to lists of their successors.
    let successors: {
      [childName: string]: string[];
    } = {};
    // Set of node names which have appeared as a destination.
    let destinations: {
      [childName: string]: boolean;
    } = {};
    let metagraph = (<GroupNode>node).metagraph;
    _.each(metagraph.edges(), (e: any) => {
      if (!metagraph.edge(e).numRegularEdges) {
        return; // Skip control edges.
      }
      // Keep track of successors and destinations.
      if (!(e.v in successors)) {
        successors[e.v] = [];
      }
      successors[e.v].push(e.w);
      destinations[e.w] = true;
    });
    // Seed the queue with true sources (those that are not destinations).
    let queue: string[] = _.difference(_.keys(successors), _.keys(destinations));
    // Produce an ordering by traversing the graph breadth first.
    this.orderings[nodeName] = {};
    let ordering = this.orderings[nodeName];
    let index = 0;
    while (queue.length) {
      let childName = queue.shift();
      if (childName) {
        ordering[childName] = index++;
        _.each(successors[childName], (succName) => queue.push(succName));
        delete successors[childName]; // Prevent cycles from infinite looping.
      }
    }
    return ordering;
  }

  /**
   * Utility function for determining the name of the immediate child under a
   * node for a given descendant path. If the descendant corresponds to no
   * immediate child, an error is thrown.
   */
  private getChildName(nodeName: string, descendantName: string): string {
    // Walk up the hierarchy from the descendant to find the child.
    let currentNode: Node | null = this.index[descendantName];
    if (!currentNode) {
      return '';
    }
    while (currentNode) {
      if (currentNode.parentNode && currentNode.parentNode.name === nodeName) {
        return currentNode.name;
      }
      currentNode = currentNode.parentNode;
    }
    throw Error(`Could not find immediate child for descendant: ${descendantName}`);
  }

  /** Helper method for getPredecessors and getSuccessors */
  private getOneWayEdges(node: GroupNode | OpNode, inEdges: boolean): tf_graph.Edges {
    let edges: Edges = { control: [], regular: [] };
    // A node with no parent cannot have any edges.
    if (!node.parentNode || !node.parentNode.isGroupNode) {
      return edges;
    }
    let parentNode = <GroupNode>node.parentNode;
    let metagraph = parentNode.metagraph;
    findEdgeTargetsInGraph(metagraph, node, inEdges, edges);
    return edges;
  }
}

/**
 * Internal utility function - given a graph (should be either a metagraph or a
 * bridgegraph) and a node which is known to be in that graph, determine
 * the other ends of edges that involve that node in the direction specified
 * by whether it's inbound.
 *
 * For example if you wanted to find the predecessors of a node, you'd call
 * this method for the parent's metagraph and bridgegraph, specifying inbound
 * as true (look at the source of inbound edges to the specified node).
 *
 * Discovered target names are appended to the targets array.
 */
function findEdgeTargetsInGraph(graph: graphlib.Graph, node: Node, inbound: boolean, targets: Edges): void {
  let edges = inbound ? graph.inEdges(node.name) : graph.outEdges(node.name);
  _.each(edges, (e) => {
    let metaedge = graph.edge(e);
    let targetList = metaedge.numRegularEdges ? targets.regular : targets.control;
    targetList.push(metaedge as any);
  });
}

export interface HierarchyParams {
  verifyTemplate: boolean;
  rankDirection: string;
}

export const DefaultHierarchyParams: HierarchyParams = {
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
      30,
      () => {
        addNodesInVis(h, graph, ROOT_NAME);
      },
      tracker,
      tb_debug.GraphDebugEventId.HIERARCHY_ADD_NODES,
    )
    .then(() => {
      return tf_graph_util.runAsyncTask(
        'Adding edges',
        70,
        () => {
          addEdgesInVis(h, graph, ROOT_NAME);
        },
        tracker,
        tb_debug.GraphDebugEventId.HIERARCHY_ADD_EDGES,
      );
    })
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
    addEdgesInVis(oldGraph, slimGraph, nodeName);
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

/**
 * Create edges in the metanode.
 * @param h
 * @param graph
 * @param nodeName
 */
function addEdgesInVis(h: Hierarchy, graph: SlimGraph, nodeName: string): void {
  const metaNode = h.node(nodeName);
  if (!(metaNode instanceof MetanodeImpl)) {
    return;
  }
  _.each(graph.edges, (baseEdge) => {
    if (!baseEdge.v || !baseEdge.w) {
      return;
    }
    if (!(baseEdge.w?.includes(tf_graph.NAMESPACE_DELIM) || baseEdge.v?.includes(tf_graph.NAMESPACE_DELIM))) {
      const isVInNodes = baseEdge.v in graph.nodes;
      const isVInMetaNodes = baseEdge.v in graph.metaNodes;
      const isWInNodes = baseEdge.w in graph.nodes;
      const isWInMetaNodes = baseEdge.w in graph.metaNodes;
      const shouldSkip = !(isVInNodes || isVInMetaNodes) || !(isWInNodes || isWInMetaNodes);
      if (shouldSkip) {
        return;
      }
    }
    const srcName = baseEdge.v;
    const dstName = baseEdge.w;
    let metaedge = metaNode.metagraph.edge(srcName, dstName, baseEdge.attr?.id);
    if (!metaedge) {
      metaedge = createMetaedge(srcName, dstName) as any;
      metaNode.metagraph.setEdge(srcName, dstName, metaedge, baseEdge.attr?.id);
    }
    metaedge.addBaseEdge(baseEdge, h, true);
  });
}