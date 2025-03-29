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
import { graphlib } from 'dagre';
import * as _ from 'lodash';
import * as tb_debug from '../tb_debug';
import { ProgressTracker } from './common';
import { Hierarchy } from './hierarchy';
import * as tf_graph_proto from './proto';
import * as tf_graph_util from './util';
import { safeJSONParse } from '../utils';

export const NAMESPACE_DELIM = '/';
export const ROOT_NAME = '__root__';
export const FUNCTION_LIBRARY_NODE_PREFIX = '__function_library__';
/** Attribute key used for storing attributes that are too large. */
export const LARGE_ATTRS_KEY = '_too_large_attrs';
/** Precision attributes are used to represent the color of nodes. */
export const NODE_TYPE = 'node_type';
export const PRECISION_INDEX = 'precision_index';
export const MATCHED_NODE_LINK = 'matched_node_link';
export const OVERFLOW_LEVEL = 'overflow_level';
/**
 * Maximum allowed size in bytes, before the attribute is considered large
 * and filtered out of the graph.
 */
export const LIMIT_ATTR_SIZE = 1024;
// Separator between the source and the destination name of the edge.
export const EDGE_KEY_DELIM = '--';
export enum GraphType {
  FULL = 0,
  EMBEDDED = 1,
  META = 2,
  SERIES = 3,
  CORE = 4,
  SHADOW = 5,
  BRIDGE = 6,
  EDGE = 7,
}
export enum NodeType {
  META = 0,
  OP = 1,
  SERIES = 2,
  BRIDGE = 3,
  ELLIPSIS = 4,
  MULTI_COLLECTION = 8,
  API_LIST = 9,
}
/** Indicates if a node is to be included in the main graph when rendered. */
export enum InclusionType {
  INCLUDE = 0,
  EXCLUDE = 1,
  UNSPECIFIED = 2,
}

// Including both the NPU and benchmark slimgraph.
export interface MergedSlimGraph {
  npu: SlimGraph;
  bench?: SlimGraph;
}
/**
 * A SlimGraph is inspired by graphlib.Graph, but having only the functionality
 * that we need.
 */
export class SlimGraph {
  nodes: {
    [nodeName: string]: OpNode;
  };
  metaNodes: {
    [nodeName: string]: Metanode;
  };
  constructor() {
    this.nodes = {};
    this.metaNodes = {};
  }
}
export interface NormalizedInput {
  name: string;
  /** The index of the output tensor of the source node. */
  outputTensorKey: string;
}
export interface BuildParams {
  enableEmbedding: boolean;
  inEmbeddingTypes: string[];
  outEmbeddingTypes: string[];
  refEdges: {
    [inputEdge: string]: boolean;
  };
}
/**
 * The most basic information about a node in the hierarchical graph.
 */
export interface Node {
  /** The name of the node, used frequently to look up nodes by name. */
  name: string;
  /** Which type of node this is. */
  type: NodeType;
  inputData: {
    [key: string]: any;
  };
  outputData: {
    [key: string]: any;
  };
  suggestions: {
    [key: string]: string;
  };
  /**
   * Whether this node is a type that may contain other nodes. Those types
   * should extend from GroupNode.
   *
   * For an OpNode, isGroupNode will be false, even though it may have
   * embeddings. These embedding Nodes will have their parentNode set to the
   * OpNode. However, embeddings are later rendered as annotations, not as
   * children to be made visible on expansion (like a Metanode or SeriesNode).
   */
  isGroupNode: boolean;
  /**
   * The number of nodes this node represents. For OpNodes, this will be 1, and
   * for GroupNodes it will be a count of the total number of descendents it
   * contains.
   */
  cardinality: number;
  /**
   * The Node which is this Node's parent. This is of type Node and not
   * GroupNode because of embeddings, which will have a parent OpNode.
   */
  parentNode: Node | null;
  /** If the node is to be included or excluded from the main graph when
   *  rendered. Defaults to UNSPECIFIED, which means that the rendering
   *  algorithm determines if it will be included or not. Then can be set to
   *  INCLUDE or EXCLUDE manually by the user.
   */
  include: InclusionType;
  /**
   * Node attributes specify customizable visual aspects of a node and
   * application-specific metadata associated with a node. The name
   * 'nodeAttributes' is meant to avoid naming-conflicts with the 'attr' in
   * subclasses of Node.
   */
  nodeAttributes: {
    [key: string]: any;
  };
}
export type TensorShape = number[];
export interface OpNode extends Node {
  op: string;
  attr: Array<{
    key: string;
    value: any;
  }>;
  inputData: {
    [key: string]: any;
  };
  outputData: {
    [key: string]: any;
  };
  stackData: [];
  matchedNodeLink: [];
  suggestions: {
    [key: string]: string;
  };
}

export interface GroupNode extends Node {
  metagraph: graphlib.Graph;
}
export interface Metanode extends GroupNode {
  depth: number;
  attr: Array<{
    key: string;
    value: any;
  }>;
  inputData: {
    [key: string]: any;
  };
  outputData: {
    [key: string]: any;
  };
  stackData: [];
  matchedNodeLink: [];
  suggestions: {
    [key: string]: string;
  };
  getFirstChild: () => GroupNode | OpNode;
  getRootOp: () => OpNode;
  /** Return name of all leaves inside a metanode. */
  leaves: () => string[];
}

/**
 * A label object for nodes in the full graph and leaf nodes in the render
 * graph.
 */
export class OpNodeImpl implements OpNode {
  name: string;
  op: string;
  attr: Array<{
    key: string;
    value: any;
  }>;
  type: NodeType;
  isGroupNode: boolean;
  cardinality: number;
  parentNode: Node | null;
  include: InclusionType;
  inputData: {
    [key: string]: any;
  };
  outputData: {
    [key: string]: any;
  };
  stackData: [];
  matchedNodeLink: [];
  suggestions: {
    [key: string]: string;
  };
  nodeAttributes: {
    [key: string]: any;
  };

  /**
   * Constructs a new Op node.
   *
   * @param rawNode The raw node.
   */
  constructor(rawNode: tf_graph_proto.NodeDef) {
    this.op = rawNode.op;
    this.name = rawNode.name;
    this.attr = rawNode.attr;
    // additional properties
    this.type = NodeType.OP;
    this.isGroupNode = false;
    this.cardinality = 1;
    this.parentNode = null;
    this.include = InclusionType.UNSPECIFIED;
    this.inputData = rawNode.input_data;
    this.outputData = rawNode.output_data;
    this.suggestions = rawNode.suggestions;
    this.stackData = rawNode.stack_info;
    this.matchedNodeLink = rawNode.matched_node_link;
    this.nodeAttributes = {};
  }
}

export function createMetanode(name: string, opt = {}): Metanode {
  return new MetanodeImpl(name, opt);
}

export class MetanodeImpl implements Metanode {
  name: string;
  type: NodeType;
  depth: number;
  isGroupNode: boolean;
  cardinality: number;
  metagraph: graphlib.Graph;
  parentNode: Node | null;
  include: InclusionType;
  inputData: {
    [key: string]: any;
  };
  outputData: {
    [key: string]: any;
  };
  stackData: [];
  matchedNodeLink: [];
  suggestions: {
    [key: string]: string;
  };
  nodeAttributes: {
    [key: string]: any;
  };
  attr: Array<{
    key: string;
    value: any;
  }>;

  /** A label object for meta-nodes in the graph hierarchy */
  constructor(name: string, opt = {}) {
    this.name = name;
    this.type = NodeType.META;
    /** number of levels under this group */
    this.depth = 1;
    this.isGroupNode = true;
    /** # of leaf nodes (including embedded ones) */
    this.cardinality = 0;
    /** graph contains metanodes, nodes, edges
     * and metaedges for main items within this metanode
     */
    this.metagraph = createGraph<GroupNode | OpNode>(name, GraphType.META, opt);
    /** Metanode which contains this node, if any */
    this.parentNode = null;
    this.include = InclusionType.UNSPECIFIED;
    this.attr = [];
    this.inputData = {};
    this.outputData = {};
    this.stackData = [];
    this.matchedNodeLink = [];
    this.suggestions = {};
    this.nodeAttributes = {};
  }

  getFirstChild(): GroupNode | OpNode {
    return this.metagraph.node(this.metagraph.nodes()[0]) as any;
  }

  /**
   * Returns the op node associated with the metanode.
   * For example, if the metanode is 'sgd', the associated
   * op node is sgd/(sgd).
   */
  getRootOp(): OpNode {
    let nameSplit = this.name.split('/');
    let rootOpName = `${this.name}/(${nameSplit[nameSplit.length - 1]})`;
    return this.metagraph.node(rootOpName) as any;
  }

  /**
   * Return an array of the names of all the leaves (non-GroupNodes) inside
   * this metanode. This performs a breadth-first search of the tree, so
   * immediate child leaves will appear earlier in the output array than
   * descendant leaves.
   */
  leaves(): string[] {
    let leaves: string[] = [];
    let queue = [this as Node];
    let metagraph; // Defined here due to a limitation of ES6->5 compilation.
    while (queue.length) {
      let node = queue.shift();
      if (node?.isGroupNode) {
        metagraph = (<GroupNode>node).metagraph;
        _.each(metagraph.nodes(), (name) => queue.push(metagraph.node(name)));
      } else {
        leaves.push(node?.name ?? '');
      }
    }
    return leaves;
  }
}

export const defaultBuildParams: BuildParams = {
  enableEmbedding: true,
  inEmbeddingTypes: ['Const'],
  outEmbeddingTypes: ['^[a-zA-Z]+Summary$'],
  // This is the whitelist of inputs on op types that are considered
  // reference edges. "Assign 0" indicates that the first input to
  // an OpNode with operation type "Assign" is a reference edge.
  refEdges: {
    'Assign 0': true,
    'AssignAdd 0': true,
    'AssignSub 0': true,
    'assign 0': true,
    'assign_add 0': true,
    'assign_sub 0': true,
    'count_up_to 0': true,
    'ScatterAdd 0': true,
    'ScatterSub 0': true,
    'ScatterUpdate 0': true,
    'scatter_add 0': true,
    'scatter_sub 0': true,
    'scatter_update 0': true,
  },
};

export function build(
  graphDef: tf_graph_proto.GraphDef,
  params: BuildParams,
  tracker?: ProgressTracker,
): Promise<SlimGraph> {
  let embeddingNodeNames: string[] = [];
  let rawNodes = graphDef.node;
  /**
   * A list of all the non-embedding node names which appear in the processed
   * list of raw nodes. Here we pre-allocate enough room for all the rawNodes,
   * even though there will some number of embeddings. The excess array length
   * is spliced off later.
   *
   * Experimentation shows that around 30% of the array will go unused, and
   * even for very large networks that amounts to less than 10k spaces.
   */
  let nodeNames = new Array<string>(rawNodes.length);
  return tf_graph_util
    .runAsyncTask(
      'Normalizing names',
      30,
      () => {
        let opNodes = new Array<OpNode | Metanode>(rawNodes.length);
        let index = 0;
        const processRawNode = (rawNode: tf_graph_proto.NodeDef): OpNodeImpl | MetanodeImpl => {
          if (!rawNode.isLeaf) {
            let metaNode = new MetanodeImpl(rawNode.name);
            metaNode.attr = rawNode.attr;
            metaNode.nodeAttributes._order = index;
            if (rawNode.matched_node_link && rawNode.matched_node_link.length > 0) {
              metaNode.nodeAttributes._linked_node = rawNode.matched_node_link;
            }
            metaNode.inputData = rawNode.input_data;
            metaNode.outputData = rawNode.output_data;
            metaNode.stackData = rawNode.stack_info;
            metaNode.matchedNodeLink = rawNode.matched_node_link;
            metaNode.suggestions = rawNode.suggestions;
            if (Number(rawNode.node_type) === 1) {
              metaNode.type = 0;
            } else {
              metaNode.type = Number(rawNode.node_type);
            }
            opNodes[index] = metaNode;
            nodeNames[index] = metaNode.name;
            index++;
            return metaNode;
          } else {
            let opNode = new OpNodeImpl(rawNode);
            opNode.nodeAttributes._order = index;
            if (rawNode.matched_node_link && rawNode.matched_node_link.length > 0) {
              opNode.nodeAttributes._linked_node = rawNode.matched_node_link;
            }
            opNodes[index] = opNode;
            nodeNames[index] = opNode.name;
            index++;
            return opNode;
          }
        };
        _.each(rawNodes, processRawNode);
        opNodes.splice(index);
        nodeNames.splice(index);
        return opNodes;
      },
      tracker,
      tb_debug.GraphDebugEventId.NORMALIZING_NAMES,
    )
    .then((opNodes) => {
      // Create the graph data structure from the graphlib library.
      return tf_graph_util.runAsyncTask(
        'Building the data structure',
        70,
        () => {
          let normalizedNameDict = mapStrictHierarchy(nodeNames, embeddingNodeNames);
          let graph = new SlimGraph();
          // Add the nodes to the graph.
          _.each(opNodes, (opNode) => {
            if (opNode instanceof OpNodeImpl) {
              let normalizedName = normalizedNameDict[opNode.name] || opNode.name;
              graph.nodes[normalizedName] = opNode;
              // Update the name of the node.
              opNode.name = normalizedName;
            } else {
              graph.metaNodes[opNode.name] = opNode as MetanodeImpl;
            }
          });
          return graph;
        },
        tracker,
        tb_debug.GraphDebugEventId.BUILD_SLIM_GRAPH,
      );
    });
}

/**
 * Create a new graphlib.Graph() instance with default parameters
 */
export function createGraph<N>(name: string, type, graphOptions: LabeledGraphOptions = {}): graphlib.Graph {
  const graph = new graphlib.Graph({ ...graphOptions, multigraph: true });
  graph.setGraph({
    name: name,
    rankdir: graphOptions.rankdir || 'TB',
    type: type,
  } as any);
  return graph;
}

/**
 * Returns a strict node name (name => name/(name)) to avoid conflicts
 * where the node name is also a namespace.
 */
export function getStrictName(name: string): string {
  let parts = name.split(NAMESPACE_DELIM);
  return `${name}${NAMESPACE_DELIM}(${parts[parts.length - 1]})`;
}

/**
 * For each op node (embedding or non-embedding), rename it if there is a
 * non-embedding node under its namespace. For example, assume node name 'A'.
 * If there is a non-embedding node under its namespace (e.g. 'A/B'), 'A' will
 * be renamed to 'A/(A)'. Then the namespace 'A' will contain 2 nodes: '(A)'
 * and 'B'. If all the nodes under 'A' are embedding nodes (e.g. constant and
 * summary), keep 'A' as an Op node and don't create a namespace.
 *
 * @param nodeNames An array of regular (non-embedding) node names.
 * @param embeddingNodeNames An array of embedding node names.
 * @return Dictionary object mapping names that need to be renamed to
 *     new names.
 */
function mapStrictHierarchy(
  nodeNames: string[],
  embeddingNodeNames: string[],
): {
  [oldName: string]: string;
} {
  /** Dictionary that maps the old new to the new name */
  let newNameDictionary: {
    [oldName: string]: string;
  } = {};
  /** Set used to store all namespaces. */
  let namespaceSet: {
    [namespace: string]: boolean;
  } = {};
  // sort the nodes to make prefix check faster
  nodeNames.sort();
  // look for nodes with a prefix a,a/b -> a/(a),a/b
  for (let i = 0; i < nodeNames.length - 1; ++i) {
    let a = nodeNames[i];
    // Get all the parent namespaces of the current node
    // and add them in the namespace set.
    _.each(getHierarchicalPath(a).slice(0, -1), (ns) => {
      namespaceSet[ns] = true;
    });
    for (let j = i + 1; j < nodeNames.length; ++j) {
      let b = nodeNames[j];
      if (_.startsWith(b, a)) {
        if (b.length > a.length && b.charAt(a.length) === NAMESPACE_DELIM) {
          newNameDictionary[a] = getStrictName(a);
          break;
        }
      } else {
        break;
      }
    }
  }
  // Go through all the embedding node names and rename them in case they
  // collide with namespaces.
  _.each(embeddingNodeNames, (embeddingName) => {
    if (embeddingName in namespaceSet) {
      // Rename to follow strict hierarchy.
      newNameDictionary[embeddingName] = getStrictName(embeddingName);
    }
  });
  return newNameDictionary;
}

/**
 * Returns the hierarchical path of the current node, based on the node's name.
 * For example, if the name is 'a/b/c', the returned path is
 * ['a', 'a/b', 'a/b/c'].
 */
export function getHierarchicalPath(name: string): string[] {
  let path: string[] = [];
  let i = name.indexOf(NAMESPACE_DELIM);
  // Push all parent portions of the path.
  while (i >= 0) {
    path.push(name.substring(0, i));
    i = name.indexOf(NAMESPACE_DELIM, i + 1);
  }
  // Push the leaf of the path.
  path.push(name);
  return path;
}

/**
 * An extended variant of the options object for `graphlib.Graph`, used
 * to configure a `graphlib.Graph` at its creation.
 *
 * Dagre's constructor has an `opts` object as a parameter, let's call it
 * 'GraphCtorOptions'. The Graph's `setGraph()` has a `label` parameter,
 * let's call it `LabelOptions`.
 *
 * Since both are configured when a `graphlib.Graph` is first initialized,
 * TensorBoard's Graph code passes around this hybrid object which includes
 * properties from both `GraphCtorOptions` (compound) and `LabelOptions`
 * (rankdir).
 */
export interface LabeledGraphOptions {
  compound?: boolean;
  rankdir?: string;
  multigraph?: boolean;
}
