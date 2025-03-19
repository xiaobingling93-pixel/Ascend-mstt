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
  createSeriesNode,
  Edges,
  getSeriesNodeName,
  GraphType,
  GroupNode,
  Metaedge,
  MetaedgeImpl,
  Metanode,
  Node,
  NodeType,
  OpNode,
  ROOT_NAME,
  SeriesNode,
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

// A map from the name of a series node to its grouping type.
type SeriesGroupMap = Map<string, tf_graph.SeriesGroupingType>;

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
  private readonly seriesGroupMap: SeriesGroupMap;

  constructor(params: HierarchyParams) {
    super();
    this.graphOptions.compound = true;
    this.graphOptions.rankdir = params.rankDirection;
    this.root = createMetanode(ROOT_NAME, this.graphOptions);
    this.seriesGroupMap = new Map(params.seriesMap);
    /**
     * @type {Object} Dictionary object that maps node name to the node
     * (could be op-node, metanode, or series-node)
     */
    this.index = {};
    this.index[ROOT_NAME] = this.root;
    this.orderings = {};
  }

  getSeriesGroupType(nodeName: string): tf_graph.SeriesGroupingType {
    // If grouping was not specified, assume it should be grouped by default.
    return this.seriesGroupMap.get(nodeName) ?? tf_graph.SeriesGroupingType.GROUP;
  }

  setSeriesGroupType(nodeName: string, groupType: tf_graph.SeriesGroupingType): SeriesGroupMap {
    return this.seriesGroupMap.set(nodeName, groupType);
  }

  buildSeriesGroupMapToggled(nodeName: string): Map<string, tf_graph.SeriesGroupingType> {
    const newGroupType =
      this.getSeriesGroupType(nodeName) === tf_graph.SeriesGroupingType.GROUP
        ? tf_graph.SeriesGroupingType.UNGROUP
        : tf_graph.SeriesGroupingType.GROUP;
    return new Map([...this.seriesGroupMap, [nodeName, newGroupType]]);
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

  /**
   * Given the name of a node in this hierarchy, get its bridgegraph, creating
   * it on the fly if necessary. If the node is not a GroupNode, then this
   * method returns null. If the provided name does not map to a node in the
   * hierarchy, an error will be thrown.
   */
  getBridgegraph(nodeName: string): graphlib.Graph {
    let node = this.index[nodeName];
    if (!node) {
      throw Error(`Could not find node in hierarchy: ${nodeName}`);
    }
    if (!('metagraph' in node)) {
      return null;
    }
    let groupNode = <GroupNode>node;
    if (groupNode.bridgegraph) {
      return groupNode.bridgegraph;
    }
    let bridgegraph = createGraph<GroupNode | OpNode, Metaedge>('BRIDGEGRAPH', GraphType.BRIDGE, this.graphOptions);
    groupNode.bridgegraph = bridgegraph;
    if (!node.parentNode || !('metagraph' in node.parentNode)) {
      return bridgegraph;
    }
    let parentNode = <GroupNode>node.parentNode;
    let parentMetagraph = parentNode.metagraph;
    let parentBridgegraph = this.getBridgegraph(parentNode.name);
    // For each of the parent node's two Metaedge containing graphs, process
    // each Metaedge involving this node.
    _.each([parentMetagraph, parentBridgegraph], (parentGraph) => {
      parentGraph
        .edges()
        .filter((e) => e.v === nodeName || e.w === nodeName)
        .forEach((parentEdgeObj) => {
          let inbound = parentEdgeObj.w === nodeName;
          let parentMetaedge = parentGraph.edge(parentEdgeObj);
          // The parent's Metaedge represents some number of underlying
          // BaseEdges from the original full graph. For each of those, we need
          // to determine which immediate child is involved and make sure
          // there's a Metaedge in the bridgegraph that covers it.
          _.each(parentMetaedge.baseEdgeList, (baseEdge) => {
            // Based on the direction, figure out which is the descendant node
            // and which is the 'other' node (sibling of parent or ancestor).
            let [descendantName, otherName] = inbound ? [baseEdge.w, parentEdgeObj.v] : [baseEdge.v, parentEdgeObj.w];
            // Determine the immediate child containing this descendant node.
            if (nodeName !== descendantName) {
              let childName = this.getChildName(nodeName, descendantName);
              if (!childName) {
                return;
              }
              // Look for an existing Metaedge in the bridgegraph (or create a
              // new one) that covers the relationship between child and other.
              let bridgeEdgeObj = <any>{
                v: inbound ? otherName : childName,
                w: inbound ? childName : otherName,
              };
              let bridgeMetaedge = bridgegraph.edge(bridgeEdgeObj);
              if (!bridgeMetaedge) {
                bridgeMetaedge = createMetaedge(bridgeEdgeObj.v, bridgeEdgeObj.w) as any;
                bridgeMetaedge.inbound = inbound;
                bridgegraph.setEdge(bridgeEdgeObj.v, bridgeEdgeObj.w, bridgeMetaedge, baseEdge.attr?.id);
              }
              // Copy the BaseEdge from the parent's Metaedge into this
              // bridgegraph Metaedge.
              bridgeMetaedge.addBaseEdge(baseEdge, this);
            }
          });
        });
    });
    return bridgegraph;
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
    let bridgegraph = this.getBridgegraph(parentNode.name);
    findEdgeTargetsInGraph(metagraph, node, inEdges, edges);
    findEdgeTargetsInGraph(bridgegraph, node, inEdges, edges);
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
  seriesNodeMinSize: number;
  // The initial map of explicit series group types.
  seriesMap: SeriesGroupMap;
  // This string is supplied to dagre as the 'rankdir' property for laying out
  // the graph. TB, BT, LR, or RL. The default is 'BT' (bottom to top).
  rankDirection: string;
  // Whether to detect numeric patterns for series nodes using entire names of
  // nodes. If false, only uses numeric suffixes to find patterns to collapse
  // into series nodes.
  useGeneralizedSeriesPatterns: boolean;
}

export const DefaultHierarchyParams: HierarchyParams = {
  verifyTemplate: true,
  seriesNodeMinSize: 5,
  seriesMap: new Map(),
  rankDirection: 'TB',
  useGeneralizedSeriesPatterns: false,
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
  const seriesNames: {
    [name: string]: string;
  } = {};
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
        'Detect series',
        30,
        () => {
          if (params.seriesNodeMinSize > 0) {
            groupSeries(
              h.root,
              h,
              seriesNames,
              params.seriesNodeMinSize,
              params.seriesMap,
              params.useGeneralizedSeriesPatterns,
            );
          }
        },
        tracker,
        tb_debug.GraphDebugEventId.HIERARCHY_DETECT_SERIES,
      );
    })
    .then(() => {
      return tf_graph_util.runAsyncTask(
        'Adding edges',
        40,
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

/**
 * Using the hierarchy template information, detect series in the provided
 * metanode.  For each detected series, create a new SeriesNode
 * and remove series members from the metanode's metagraph and move them to
 * the new series node's metagraph.
 *
 * @param metanode
 * @param hierarchy
 * @param seriesNames Map of node names to their series they are contained in.
 *     This should be provided empty and is populated by this method.
 * @param threshold If the series has this many nodes or more, then group them
 *     into a series.
 * @param map Map of series names to their series grouping type, if one has
 *     been set.
 * @param useGeneralizedSeriesPatterns Whether to use find patterns for series
 *     nodes using any parts of names of nodes. If false, only uses patterns
 *     discovered within numeric suffixes of nodes names.
 * @return A dictionary from node name to series node name that contains the
 *     node.
 */
function groupSeries(
  metanode: Metanode,
  hierarchy: Hierarchy,
  seriesNames: {
    [name: string]: string;
  },
  threshold: number,
  seriesMap: SeriesGroupMap,
  useGeneralizedSeriesPatterns: boolean,
): void {
  let metagraph = metanode.metagraph;
  _.each(metagraph.nodes(), (n) => {
    let child = metagraph.node(n);
    if (child.type === (tf_graph.NodeType.META || tf_graph.NodeType.MULTI_COLLECTION || tf_graph.NodeType.API_LIST)) {
      groupSeries(
        child as unknown as Metanode,
        hierarchy,
        seriesNames,
        threshold,
        seriesMap,
        useGeneralizedSeriesPatterns,
      );
    }
  });
  let clusters = clusterNodes(metagraph);
  const detectSeriesMethod = useGeneralizedSeriesPatterns
    ? detectSeriesAnywhereInNodeName
    : detectSeriesUsingNumericSuffixes;
  let seriesDict = detectSeriesMethod(clusters, metagraph, hierarchy.graphOptions);
  // Add each series node to the graph and add its grouped children to its own
  // metagraph.
  _.each(seriesDict, function (seriesNode: SeriesNode, seriesName: string) {
    let nodeMemberNames = seriesNode.metagraph.nodes();
    _.each(nodeMemberNames, (n) => {
      let child = metagraph.node(n) as any;
      if (!child.owningSeries) {
        child.owningSeries = seriesName;
      }
    });
    // If the series contains less than the threshold number of nodes, then set
    // this series to be shown ungrouped in the map.
    if (
      nodeMemberNames.length < threshold &&
      hierarchy.getSeriesGroupType(seriesNode.name) === tf_graph.SeriesGroupingType.GROUP
    ) {
      hierarchy.setSeriesGroupType(seriesNode.name, tf_graph.SeriesGroupingType.UNGROUP);
    }
    // If the series is in the map as ungrouped then do not group the series.
    if (hierarchy.getSeriesGroupType(seriesNode.name) === tf_graph.SeriesGroupingType.UNGROUP) {
      return;
    }
    hierarchy.setNode(seriesName, seriesNode); // add to the index
    metagraph.setNode(seriesName, seriesNode);
    _.each(nodeMemberNames, (n) => {
      let child = metagraph.node(n) as any;
      seriesNode.metagraph.setNode(n, child);
      seriesNode.parentNode = child.parentNode;
      seriesNode.cardinality++;
      child.parentNode = seriesNode;
      seriesNames[n] = seriesName;
      // Remove now-grouped node from its original parent's metagraph.
      metagraph.removeNode(n);
    });
  });
}

/**
 * Cluster op-nodes with similar op. This examines only the direct children of
 * the metagraph, does not recursively check descendants.
 * @return A map from op to a list of node names.
 */
function clusterNodes(metagraph: graphlib.Graph): {
  [clusterId: string]: string[];
} {
  let result: {
    [clusterId: string]: string[];
  } = {};
  return _.reduce(
    metagraph.nodes(),
    (
      clusters: {
        [clusterId: string]: string[];
      },
      n: string,
    ) => {
      let child = metagraph.node(n);
      if (child.type === (NodeType.META || NodeType.MULTI_COLLECTION || NodeType.API_LIST)) {
        // skip metanodes
        return clusters;
      }
      let template = (child as any).op;
      if (template) {
        clusters[template] = clusters[template] || [];
        clusters[template].push(child.name);
      }
      return clusters;
    },
    result,
  );
}

/**
 * For each cluster of op-nodes based op type, try to detect groupings.
 * Infer series name using by trying to find pattern '<number>' towards the end
 * of node names.
 *
 * @param clusters Dictionary output from clusterNodes().
 * @param metagraph
 * @return A dictionary from series name => seriesNode
 */
function detectSeriesUsingNumericSuffixes(
  clusters: {
    [clusterId: string]: string[];
  },
  metagraph: graphlib.Graph,
  graphOptions: tf_graph.LabeledGraphOptions,
): {
  [seriesName: string]: SeriesNode;
} {
  let seriesDict: {
    [seriesName: string]: SeriesNode;
  } = {};
  _.each(clusters, function (members, clusterId: string) {
    if (members.length <= 1) {
      return;
    } // isolated clusters can't make series
    /** @type {Object}  A dictionary mapping seriesName to seriesInfoArray,
     * which is an array that contains objects with name, id, prefix, suffix,
     * and parent properties.
     */
    let candidatesDict: {
      [seriesName: string]: SeriesNode[];
    } = {};
    // Group all nodes that have the same name, with the exception of a
    // number at the end of the name after an underscore, which is allowed to
    // vary.
    _.each(members, function (name: string) {
      const isGroup = name.charAt(name.length - 1) === '*';
      const namepath = name.split('/');
      const leaf = namepath[namepath.length - 1];
      const parent = namepath.slice(0, namepath.length - 1).join('/');
      const matches = leaf.match(/^((?<nonDigits>\D*))(\d+)$/);

      let prefix;
      let id;
      let suffix = '';
      if (matches) {
        // if found '<number>' in the name, assign id.
        prefix = matches[1]; // the front non-numeric characters
        id = matches[2]; // the digits
      } else {
        // for node without '_<number>', make them zero-th items.
        prefix = isGroup ? leaf.substr(0, leaf.length - 1) : leaf;
        id = 0;
        suffix = isGroup ? '*' : '';
      }
      const seriesName = getSeriesNodeName(prefix, suffix, parent);
      candidatesDict[seriesName] = candidatesDict[seriesName] || [];
      const seriesNode = createSeriesNode(prefix, suffix, parent, Number(id), name, graphOptions);
      candidatesDict[seriesName].push(seriesNode);
    });
    // In each group of nodes, group nodes in bunches that have monotonically
    // increasing numbers in their names.  Each of these bunches is a series.
    _.each(candidatesDict, function (seriesInfoArray: SeriesNode[], seriesName) {
      if (seriesInfoArray.length < 2) {
        return;
      }
      seriesInfoArray.sort(function (a, b) {
        return Number(a.clusterId) - Number(b.clusterId);
      });
      // Loop through the nodes sorted by its detected series number, grouping
      // all nodes with monotonically-increasing series numbers.
      let seriesNodes = [seriesInfoArray[0]];
      for (let index = 1; index < seriesInfoArray.length; index++) {
        let nextNode = seriesInfoArray[index];
        if (nextNode.clusterId === seriesNodes[seriesNodes.length - 1].clusterId + 1) {
          seriesNodes.push(nextNode);
          continue;
        }
        addSeriesToDict(seriesNodes, seriesDict, Number(clusterId), metagraph, graphOptions);
        seriesNodes = [nextNode];
      }
      addSeriesToDict(seriesNodes, seriesDict, Number(clusterId), metagraph, graphOptions);
    });
  });
  return seriesDict;
}

/**
 * For each cluster of op-nodes based op type, try to detect groupings.
 * Infer series name using by trying to find a pattern of numbers
 * anywhere within node names.
 *
 * @param clusters Dictionary output from clusterNodes().
 * @param metagraph
 * @return A dictionary from series name => seriesNode
 */
function detectSeriesAnywhereInNodeName(
  clusters: {
    [clusterId: string]: string[];
  },
  metagraph: graphlib.Graph,
  graphOptions: tf_graph.LabeledGraphOptions,
): {
  [seriesName: string]: SeriesNode;
} {
  let seriesDict: {
    [seriesName: string]: SeriesNode;
  } = {};
  _.each(clusters, function (members, clusterId: string) {
    if (members.length <= 1) {
      return;
    } // isolated clusters can't make series
    /**
     * @type {Object}  A dictionary mapping a series name to a SeriesNode.
     */
    let forwardDict: {
      [seriesName: string]: SeriesNode;
    } = {};
    /**
     * @type {Object}  A dictionary mapping member name to an array of series
     * names this member could potentially be grouped under and the
     * corresponding ids.
     */
    let reverseDict: {
      [seriesName: string]: any[];
    } = {};
    // Group all nodes that have the same name, with the exception of a
    // number at the end of the name after an underscore, which is allowed to
    // vary.
    _.each(members, function (name: string) {
      let isGroup = name.charAt(name.length - 1) === '*';
      let namepath = name.split('/');
      let leaf = namepath[namepath.length - 1];
      let parent = namepath.slice(0, namepath.length - 1).join('/');
      const numRegex = /(?<number>\d+)/g;
      let matches = [];
      let matchResult;
      let prefix;
      let id;
      let suffix;
      let seriesName;
      let matched = 0;
      // Scan over the entire leaf name and match any possible numbers,
      // and put the results into corresponding dictionaries.
      while ((matchResult = numRegex.exec(leaf))) {
        ++matched;
        prefix = leaf.slice(0, matchResult.index);
        id = matchResult[0];
        suffix = leaf.slice(matchResult.index + matchResult[0].length);
        seriesName = getSeriesNodeName(prefix, suffix, parent);
        forwardDict[seriesName] = forwardDict[seriesName];
        if (!forwardDict[seriesName]) {
          forwardDict[seriesName] = createSeriesNode(prefix, suffix, parent, Number(id), name, graphOptions);
        }
        forwardDict[seriesName].ids.push(id);
        reverseDict[name] = reverseDict[name] || [];
        reverseDict[name].push([seriesName, id]);
      }
      if (matched < 1) {
        prefix = isGroup ? leaf.substr(0, leaf.length - 1) : leaf;
        id = 0;
        suffix = isGroup ? '*' : '';
        seriesName = getSeriesNodeName(prefix, suffix, parent);
        forwardDict[seriesName] = forwardDict[seriesName];
        if (!forwardDict[seriesName]) {
          forwardDict[seriesName] = createSeriesNode(prefix, suffix, parent, Number(id), name, graphOptions);
        }
        forwardDict[seriesName].ids.push(id);
        reverseDict[name] = reverseDict[name] || [];
        reverseDict[name].push([seriesName, id]);
      }
    });
    /** @type {Object}  A dictionary mapping seriesName to seriesInfoArray,
     * which is an array that contains objects with name, id, prefix, suffix,
     * and parent properties.
     */
    let candidatesDict: {
      [seriesName: string]: SeriesNode[];
    } = {};
    // For each of the member, put it into the maximum possible series,
    // and create candidatesDict accordingly.
    _.each(reverseDict, function (seriesNameIdArray, name) {
      seriesNameIdArray.sort(function (a, b) {
        return forwardDict[b[0]].ids.length - forwardDict[a[0]].ids.length;
      });
      let seriesName = seriesNameIdArray[0][0];
      let id = seriesNameIdArray[0][1];
      candidatesDict[seriesName] = candidatesDict[seriesName] || [];
      const namepath = name.split('/');
      const leaf = namepath[namepath.length - 1];
      const parent = namepath.slice(0, namepath.length - 1).join('/');
      let seriesNode = createSeriesNode(
        forwardDict[seriesName].prefix,
        forwardDict[seriesName].suffix,
        parent,
        Number(id),
        name,
        graphOptions,
      );
      candidatesDict[seriesName].push(seriesNode);
    });
    // In each group of nodes, group nodes in bunches that have monotonically
    // increasing numbers in their names.  Each of these bunches is a series.
    _.each(candidatesDict, (seriesInfoArray: SeriesNode[], seriesName) => {
      if (seriesInfoArray.length < 2) {
        return;
      }
      seriesInfoArray.sort((a, b) => {
        return Number(a.clusterId) - Number(b.clusterId);
      });
      // Loop through the nodes sorted by its detected series number, grouping
      // all nodes with monotonically-increasing series numbers.
      let seriesNodes = [seriesInfoArray[0]];
      for (let index = 1; index < seriesInfoArray.length; index++) {
        let nextNode = seriesInfoArray[index];
        if (nextNode.clusterId === seriesNodes[seriesNodes.length - 1].clusterId + 1) {
          seriesNodes.push(nextNode);
          continue;
        }
        addSeriesToDict(seriesNodes, seriesDict, Number(clusterId), metagraph, graphOptions);
        seriesNodes = [nextNode];
      }
      addSeriesToDict(seriesNodes, seriesDict, Number(clusterId), metagraph, graphOptions);
    });
  });
  return seriesDict;
}

/**
 * Add a series to the provided dictionary mapping series names to series.
 *
 * @param seriesNodes the nodes in the series. Contains
 *     name, id, prefix, suffix and parent properties of the node.
 * @param seriesDict the dictionary of series
 * @param clusterId ID of the template of the nodes of the series
 * @param metagraph
 * @param graphOptions
 */
function addSeriesToDict(
  seriesNodes: SeriesNode[],
  seriesDict: {
    [seriesName: string]: SeriesNode;
  },
  clusterId: number,
  metagraph: graphlib.Graph,
  graphOptions: tf_graph.LabeledGraphOptions,
): void {
  if (seriesNodes.length > 1) {
    let curSeriesName = getSeriesNodeName(
      seriesNodes[0].prefix,
      seriesNodes[0].suffix,
      seriesNodes[0].parent,
      seriesNodes[0].clusterId,
      seriesNodes[seriesNodes.length - 1].clusterId,
    );
    let curSeriesNode = createSeriesNode(
      seriesNodes[0].prefix,
      seriesNodes[0].suffix,
      seriesNodes[0].parent,
      clusterId,
      curSeriesName,
      graphOptions,
    );
    _.each(seriesNodes, (node) => {
      curSeriesNode.ids.push(node.clusterId);
      curSeriesNode.metagraph.setNode(node.name, metagraph.node(node.name));
    });
    seriesDict[curSeriesName] = curSeriesNode;
  }
}
