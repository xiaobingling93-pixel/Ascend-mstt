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
 * Package for the Render Hierarchy for TensorFlow graph.
 */
import * as d3 from 'd3';
import { graphlib } from 'dagre';
import * as _ from 'lodash';
import { NPU_PREFIX, BENCH_PREFIX, EDGE_WIDTH_SIZE_BASED_SCALE } from './common';
import * as tf_graph from './graph';
import {
  BridgeNode,
  createGraph,
  EllipsisNode,
  getHierarchicalPath,
  GraphType,
  GroupNode,
  InclusionType,
  Metaedge,
  MetaedgeImpl,
  Metanode,
  Node,
  NodeType,
  OpNode,
  OpNodeImpl,
} from './graph';
import { Hierarchy } from './hierarchy';
import { NodeOpType } from './proto';

const NODE_LINE_FEED_NUMBER = 5;
export interface EdgeData {
  v: string;
  w: string;
  id: number;
  label: RenderMetaedgeInfo;
}

export interface Point {
  x: number;
  y: number;
}
/**
 * Color parameters for op nodes.
 */
export const OpNodeColors = {
  DEFAULT_FILL: '#ffffff',
  DEFAULT_STROKE: '#b2b2b2',
};
/**
 * Color parameters for node encoding.
 * @type {Object}
 */
export let MetanodeColors = {
  /**
   * Default fill and stroke to use when no other information is available.
   */
  DEFAULT_FILL: '#d9d9d9',
  DEFAULT_STROKE: '#a6a6a6',
  SATURATION: 0.6,
  LIGHTNESS: 0.85,
  /**
   * Neutral color to use when the node is expanded (used when coloring by
   * compute time, memory and device).
   */
  EXPANDED_COLOR: '#f0f0f0',
  /**
   * Standard hue values for node color palette.
   */
  HUES: [220, 100, 180, 40, 20, 340, 260, 300, 140, 60],
  STRUCTURE_PALETTE(id: number, lightened?: boolean): string {
    // The code below is a flexible way to computationally create a set
    // of colors that go well together.
    let hues = MetanodeColors.HUES;
    let n = hues.length;
    let hue = hues[id % n];
    let m = Math.sin((hue * Math.PI) / 360);
    let sat = lightened ? 30 : 90 - 60 * m;
    let light = lightened ? 95 : 80;
    return d3.hsl(hue, 0.01 * sat, 0.01 * light).toString();
  },
  DEVICE_PALETTE(index: number): string {
    return MetanodeColors.STRUCTURE_PALETTE(index);
  },
  XLA_CLUSTER_PALETTE(index: number): string {
    return MetanodeColors.STRUCTURE_PALETTE(index);
  },
  UNKNOWN: '#eee',
  GRADIENT_OUTLINE: '#888',
};
/**
 * Color parameters for op nodes.
 */
export const SeriesNodeColors = {
  DEFAULT_FILL: 'white',
  DEFAULT_STROKE: '#b2b2b2',
};
/**
 * Function that computes edge thickness in pixels.
 */
export interface EdgeThicknessFunction {
  (edgeData: EdgeData, edgeClass: string): number;
}
/**
 * Function that computes edge label strings. This function accepts a Metaedge,
 * which could actually encapsulate several base edges. For instance, several
 * base edges may merge into a single metaedge.
 *
 * To determine whether a metaedge represents several edges, check the length of
 * its baseEdgeList property.
 */
export interface EdgeLabelFunction {
  (metaedge: Metaedge, renderInfo: RenderGraphInfo): string;
}
/**
 * Parameters that affect how the graph is rendered on the screen.
 */
const PARAMS = {
  /**
   * The minimum number of nodes for a graph to have in order for high in and
   * out degree nodes to be extracted in auxiliary. The aim here is to prevent
   * nodes from being extracted from small graphs.
   */
  minNodeCountForExtraction: 15,
  /**
   * The minimum in or out degree a node must have in order to be possibly
   * extracted.
   */
  minDegreeForExtraction: 5,
  /**
   * Maximum number of control edges a node can have before they aren't
   * displayed.
   */
  maxControlDegree: 4,
  /**
   * Maximum in (for outbound bridge paths) or out (for inbound bridge paths)
   * degree of a node allowed for a bridge path to be rendered to it from a
   * subhierarchy of nodes. Having a max prevents having too many nodes emanate
   * from a subhierarchy and crowding up.
   */
  maxBridgePathDegree: 4,
  /**
   * Types patterns for predefined out-extract nodes, which are
   * sink-like nodes that will be extracted from the main graph.
   */
  outExtractTypes: ['NoOp'],
  /**
   * Types patterns for predefined in-extract nodes, which are
   * source-like nodes that will be extracted from the main graph.
   */
  inExtractTypes: [],
  /**
   * When removing edges from a high degree node, remove all of its edges if
   * detachAllEdgesForHighDegree is true.  Otherwise remove all in-edges if
   * the node has high in-degree, or all out-edges if the node has high
   * out-degree.
   */
  detachAllEdgesForHighDegree: true,
  /**
   * After extracting high in/out degree nodes and predefined
   * source-like/sink-like, extract isolated nodes to the side
   * if this extractIsolatedNodesWithAnnotationsOnOneSide is true.
   */
  extractIsolatedNodesWithAnnotationsOnOneSide: true,
  /**
   * Whether to add bridge nodes and edges to the core when building the
   * subhierarchy of an expanded metanode. See buildSubhierarchy().
   */
  enableBridgegraph: true,
  /**
   * 2 colors, for the minimum and maximum value respectively, whenever we
   * have a gradient scale.
   */
  minMaxColors: ['#fff5f0', '#fb6a4a'],
  /**
   * Maximum number of annotations to be displayed on a node before an
   * ellipsis is used.
   */
  maxAnnotations: 5,
};
/**
 * The regular expression to use when parsing for the string that is
 * used to label a function node in the graph. We strip away a prefix
 * indicating that the node represents a function definition. We also
 * remove an arbitrary hexadecimal suffix and the number following it
 * if it is present. To be clear, we extract foo from
 * __function_library__foo_deadb00f_42.
 */
const nodeDisplayNameRegex = new RegExp(
  `^(?:'${tf_graph.FUNCTION_LIBRARY_NODE_PREFIX}')?(\\w+)_[a-z0-9]{8}(?:_\\d+)?$`,
);
// 同时包含npu侧和标杆侧的图渲染信息
export interface MergedRenderGraphInfo {
  npu: RenderGraphInfo;
  bench?: RenderGraphInfo;
}
/**
 * Stores the rendering information, such as x and y coordinates,
 * for each node in the graph.
 */
export class RenderGraphInfo {
  hierarchy: Hierarchy;
  renderedOpNames: string[];
  /** Scale for the thickness of edges when there is no shape information. */
  edgeWidthSizedBasedScale: d3.ScaleLinear<number, number> | d3.ScalePower<number, number>;
  root: RenderGroupNodeInfo;
  traceInputs: boolean;
  private index: {
    [nodeName: string]: RenderNodeInfo;
  };
  private toRenderEdgeObjs: {
    v: string;
    w: string;
    id: number;
    edge: tf_graph.Metaedge;
  }[];
  // Since the rendering information for each node is constructed lazily,
  // upon node's expansion by the user, we keep a map between the node's name
  // and whether the rendering information was already constructed for that
  // node.
  private hasSubhierarchy: {
    [nodeName: string]: boolean;
  };
  constructor(hierarchy: Hierarchy) {
    this.hierarchy = hierarchy;
    this.index = {};
    this.toRenderEdgeObjs = [];
    this.renderedOpNames = [];
    // Maps node name to whether the rendering hierarchy was already
    // constructed.
    this.hasSubhierarchy = {};
    this.root = new RenderGroupNodeInfo(hierarchy.root, hierarchy.graphOptions);
    this.index[hierarchy.root.name] = this.root;
    this.renderedOpNames.push(hierarchy.root.name);
    this.buildSubhierarchy(hierarchy.root.name);
    this.root.expanded = true;
    this.traceInputs = false;
    this.edgeWidthSizedBasedScale = EDGE_WIDTH_SIZE_BASED_SCALE;
  }

  /**
   * Get index.
   */
  getIndex(): { [nodeName: string]: RenderNodeInfo } {
    return this.index;
  }

  /**
   * Get a previously created RenderNodeInfo by its node name.
   */
  getRenderNodeByName(nodeName: string): RenderNodeInfo {
    return this.index[nodeName];
  }

  /**
   * Get the underlying node in the hierarchical graph by its name.
   */
  getNodeByName(nodeName: string): Node {
    return this.hierarchy.node(nodeName);
  }

  /**
   * Get a previously created RenderNodeInfo for the specified node name,
   * or create one if it hasn't been created yet.
   */
  getOrCreateRenderNodeByName(nodeName: string): RenderNodeInfo | null {
    // Polymer may invoke this with null.
    if (!nodeName) {
      return null;
    }
    if (nodeName in this.index) {
      return this.index[nodeName];
    }
    let node = this.hierarchy.node(nodeName);
    // Exit early if the node does not exist in the hierarchy. This can happen
    // when a graph is reloaded while the infocard points to a node not visible
    // at the top-level.
    if (!node) {
      return null;
    }
    let renderInfo = node.isGroupNode
      ? new RenderGroupNodeInfo(<GroupNode>node, this.hierarchy.graphOptions)
      : new RenderNodeInfo(node);
    this.index[nodeName] = renderInfo;
    this.renderedOpNames.push(nodeName);
    return this.index[nodeName];
  }

  /**
   * Return the nearest ancestor node, including itself, that is visible
   * in the visualization. This method is used so that we can select
   * (highlight) a node that isn't drawn yet, by selecting (highlighting)
   * its nearest ancestor that has been drawn.
   */
  getNearestVisibleAncestor(name: string): string {
    let path = getHierarchicalPath(name);
    let i = 0;
    let renderNode: RenderNodeInfo | null = null;
    // Fallthrough. If everything was expanded return the node.
    let nodeName = name;
    for (; i < path.length; i++) {
      nodeName = path[i];
      renderNode = this.getRenderNodeByName(nodeName);
      // Op nodes have expanded set to false by default.
      if (renderNode && !renderNode.expanded) {
        break;
      }
    }
    if (!renderNode) {
      return '';
    }
    // Check case where highlighted node is an embedded node whose parent node
    // is also its hierarchical parent. In this case, we want to return the
    // embedded node name, as it is also displayed if its parent has been
    // displayed.
    if (i === path.length - 2) {
      let nextName = path[i + 1];
      if (renderNode?.inAnnotations.nodeNames[nextName] || renderNode?.outAnnotations.nodeNames[nextName]) {
        return nextName;
      }
    }
    return nodeName;
  }

  setDepth(depth: number): void {
    setGroupNodeDepth(this.root, Number(depth));
  }

  /**
   * Returns true if the renderNode is an isolated node within its parent node.
   */
  isNodeAuxiliary(renderNode: RenderNodeInfo): boolean {
    let parentNode = <RenderGroupNodeInfo>this.getRenderNodeByName(renderNode.node.parentNode.name);
    let found = _.find(parentNode.isolatedInExtract, (node) => {
      return node.node.name === renderNode.node.name;
    });
    if (found) {
      return true;
    }
    found = _.find(parentNode.isolatedOutExtract, (node) => {
      return node.node.name === renderNode.node.name;
    });
    return !!found;
  }

  /**
   * Returns a list of ops that have been rendered so far for this graph. More
   * ops may later be rendered if the user expands nodes for instance. The list
   * returned here can only stay the same size or grow on successive calls.
   */
  getNamesOfRenderedOps(): string[] {
    return this.renderedOpNames;
  }

  buildSubhierarchy(nodeName: string, subGraph: tf_graph.SlimGraph | undefined = undefined): void {
    // Terminate if the rendering hierarchy was already constructed
    // for this node.
    if (nodeName in this.hasSubhierarchy) {
      return;
    }
    // Record that we constructed the rendering hierarchy for this node, so we
    // don't construct it another time.
    this.hasSubhierarchy[nodeName] = true;
    let renderNodeInfo = this.index[nodeName];
    // If it is not a meta node or a series node, don't do anything.
    const excludedTypes = [NodeType.META, NodeType.MULTI_COLLECTION, NodeType.API_LIST, NodeType.SERIES];
    if (!excludedTypes.includes(renderNodeInfo.node.type)) {
      return;
    }
    // At this point we know the rendering information is about a group node.
    let renderGroupNodeInfo = <RenderGroupNodeInfo>renderNodeInfo;
    let metagraph = renderGroupNodeInfo.node.metagraph;
    let coreGraph = renderGroupNodeInfo.coreGraph;
    // Create render nodes to represent each child from the metagraph. Although
    // these will initially be added to the coreGraph, they may later be
    // extracted. Also, due to extraction, the coreGraph may contain disjoint
    // groups between which there is no visible path (other than annotations).
    _.each(metagraph.nodes(), (childName, index: number) => {
      let childRenderInfo = this.getOrCreateRenderNodeByName(childName);
      if (!childRenderInfo) {
        return;
      }
      let childNode = childRenderInfo.node;
      coreGraph.setNode(childName, childRenderInfo);
      // 可展开节点自成一行，模块间游离节点每NODE_LINE_FEED_NUMBER个换行
      if (index >= 1 && subGraph && Object.keys(subGraph.metaNodes).length > 0) {
        coreGraph.setEdge(metagraph.nodes()[index - 1], childName, {});
      } else if (index >= NODE_LINE_FEED_NUMBER && subGraph && Object.keys(subGraph.metaNodes).length === 0) {
        coreGraph.setEdge(metagraph.nodes()[index - NODE_LINE_FEED_NUMBER], childName, {});
      }
      if (!childNode.isGroupNode) {
        _.each((<OpNode>childNode).inEmbeddings, (embedding) => {
          let renderMetaedgeInfo = new RenderMetaedgeInfo(null);
          let renderNodeData = new RenderNodeInfo(embedding);
          addInAnnotation(
            childRenderInfo as RenderNodeInfo,
            embedding,
            renderNodeData,
            renderMetaedgeInfo,
            AnnotationType.CONSTANT,
          );
          this.index[embedding.name] = renderNodeData;
        });
        _.each((<OpNode>childNode).outEmbeddings, (embedding) => {
          let renderMetaedgeInfo = new RenderMetaedgeInfo(null);
          let renderNodeData = new RenderNodeInfo(embedding);
          addOutAnnotation(
            childRenderInfo as RenderNodeInfo,
            embedding,
            renderNodeData,
            renderMetaedgeInfo,
            AnnotationType.SUMMARY,
          );
          this.index[embedding.name] = renderNodeData;
        });
      }
    });
    // Add render metaedge info for edges in the metagraph.
    _.each(metagraph.edges(), (edgeObj) => {
      if (edgeObj.v.includes(tf_graph.NAMESPACE_DELIM) || edgeObj.w.includes(tf_graph.NAMESPACE_DELIM)) {
        let inbound = edgeObj.w.includes(tf_graph.NAMESPACE_DELIM);
        if (inbound) {
          let pathNames = edgeObj.w.split(tf_graph.NAMESPACE_DELIM);
          this.toRenderEdgeObjs.push({
            v: edgeObj.v,
            w: pathNames[pathNames.length - 1],
            id: edgeObj.name,
            edge: metagraph.edge(edgeObj),
          });
        } else {
          let pathNames = edgeObj.v.split(tf_graph.NAMESPACE_DELIM);
          this.toRenderEdgeObjs.push({
            v: pathNames[pathNames.length - 1],
            w: edgeObj.w,
            id: edgeObj.name,
            edge: metagraph.edge(edgeObj),
          });
        }
        return;
      }
      let metaedge = metagraph.edge(edgeObj);
      let renderMetaedgeInfo = new RenderMetaedgeInfo(metaedge as any);
      renderMetaedgeInfo.isFadedOut = this.index[edgeObj.v].isFadedOut || this.index[edgeObj.w].isFadedOut;
      coreGraph.setEdge(edgeObj.v, edgeObj.w, renderMetaedgeInfo, edgeObj.name);
    });
    // Look up the parent node's render information and short circuit if none.
    let parentNode = renderGroupNodeInfo.node.parentNode;
    if (!parentNode) {
      return;
    }
    let parentNodeInfo = <RenderGroupNodeInfo>this.index[parentNode.name];
    // Utility function for computing the name of a bridge node.
    let getBridgeNodeName = (inbound, ...rest): string => rest.concat([inbound ? 'IN' : 'OUT']).join('~~');
    if (subGraph) {
      const subNodes = Object.keys(subGraph.metaNodes).concat(Object.keys(subGraph.nodes));
      renderNodeInfo.node.cardinality += subNodes.length;
      const parentMetagraph = (parentNode as GroupNode).metagraph;
      const parentBridgegraph = (parentNode as GroupNode).bridgegraph;
      _.each(subNodes, (subName) => {
        this.toRenderEdgeObjs
          .filter((e) => e.v === subName || e.w === subName)
          .forEach((item) => {
            const edgeObj = item.edge;
            const inbound = item.w === subName;
            const edgeIO = inbound
              ? [edgeObj.v?.split(tf_graph.NAMESPACE_DELIM)[0], nodeName]
              : [nodeName, edgeObj.w?.split(tf_graph.NAMESPACE_DELIM)[0]];
            let bridgeMetaedge = parentMetagraph.edge(...edgeIO, item.id);
            if (!bridgeMetaedge) {
              bridgeMetaedge = parentBridgegraph.edge(...edgeIO, item.id);
              if (!bridgeMetaedge) {
                return;
              }
            }
            _.each(edgeObj.baseEdgeList, (baseEdge) => {
              let name = inbound ? baseEdge.v : baseEdge.w;
              if (baseEdge.attr) {
                baseEdge.attr._path = name;
              } else {
                baseEdge.attr = { _path: name };
              }
              if (inbound) {
                baseEdge.v = name?.split(tf_graph.NAMESPACE_DELIM).pop();
                baseEdge.w = subName;
              } else {
                baseEdge.v = subName;
                baseEdge.w = name?.split(tf_graph.NAMESPACE_DELIM).pop();
              }
              bridgeMetaedge.addBaseEdge(baseEdge, this.hierarchy, true);
            });
          });
      });
    }
    // Remove rendered edges to save memory.
    this.toRenderEdgeObjs = this.toRenderEdgeObjs.filter((e) => !this.index[e.v] || !this.index[e.w]);
    // Build out the bridgegraph.
    let bridgegraph = this.hierarchy.getBridgegraph(nodeName);
    // Look for popular nodes so we can make annotations instead of paths.
    let otherCounts = {
      // Counts of edges coming INTO other nodes by name (outgoing from self).
      in: <
        {
          [nodeName: string]: number;
        }
      >{},
      // Counts of edges going OUT from other nodes by name (coming into self).
      out: <
        {
          [nodeName: string]: number;
        }
      >{},
      // Counts of all control edges involving other nodes by name.
      control: <
        {
          [nodeName: string]: number;
        }
      >{},
    };
    _.each(bridgegraph.edges(), (e) => {
      // An edge is inbound if its destination node is in the metagraph.
      let inbound = !!metagraph.node(e.w);
      let otherName = inbound ? e.v : e.w;
      let metaedge = bridgegraph.edge(e);
      if (!metaedge.numRegularEdges) {
        otherCounts.control[otherName] = (otherCounts.control[otherName] || 0) + 1;
      } else if (inbound) {
        otherCounts.out[otherName] = (otherCounts.out[otherName] || 0) + 1;
      } else {
        otherCounts.in[otherName] = (otherCounts.in[otherName] || 0) + 1;
      }
    });
    // For each bridge container (IN and/or OUT), add structural edges between
    // terminal nodes and that container. A terminal node is one which has no
    // non-bridge edges in the direction of the container.
    //
    // For example, consider a Metanode A which contains two child nodes A/B
    // and A/C. Let's say it has one edge in the metagraph from A/B->A/C, and
    // one edge in the bridgegraph from Z->A/C.
    //
    // At this point, we've added a container bridge node IN to house all
    // incoming bridge nodes. We've also added a bridge node Z' (with parent IN)
    // to A, and a bridge edge from Z'->C.
    //
    //     +----------------------+
    //     | A          +---+     |
    //     |    +------>| C |     |
    //     |    |       +---+     |
    //     |    |         ^       |
    //     |    |         |       |
    //     |    |    +----|----+  |
    //     |    |    | IN |    |  |
    //     |  +---+  |  +---+  |  |
    //     |  | B |  |  | Z'|  |  |
    //     |  +---+  |  +---+  |  |
    //     |         +---------+  |
    //     +----------------------+
    //
    // With no other help, dagre would lay out B and Z' on the same level,
    // because both of them have no incoming edges. In other words, B is a
    // terminal node in the INCOMING direction.
    //
    // But we want to force dagre to lay out Z' (and everything in IN) lower
    // than all non-bridge nodes, so that there's enough room for the bridge
    // edges after they've been adjusted to meet up with paths coming in from
    // outside.
    //
    // To force Z' (and all other bridge nodes) to be lowest in the graph, we
    // identify terminal nodes like B and give them structural edges to
    // a new structural bridge node S which we add to IN.
    //
    //     +----------------------+
    //     | A          +---+     |
    //     |       +--->| C |     |
    //     |       |    +---+     |
    //     |     +---+    ^       |
    //     |     | B |    |       |
    //     |     +---+    |       |
    //     |       ^      |       |
    //     |       |      |       |
    //     |  +----|------|----+  |
    //     |  |IN  |      |    |  |
    //     |  |  +---+  +---+  |  |
    //     |  |  | S |  | Z'|  |  |
    //     |  |  +---+  +---+  |  |
    //     |  +----------------+  |
    //     +----------------------+
    //
    // This ensures that dagre will lay out the bridge containers strictly at
    // the ends of the graph. The structural edges will never be seen in the
    // visualization except as a debugging aid.
    _.each([true, false], (inbound) => {
      let bridgeContainerName = getBridgeNodeName(inbound, nodeName);
      let bridgeContainerInfo = coreGraph.node(bridgeContainerName);
      if (!bridgeContainerInfo) {
        return;
      }
      _.each(coreGraph.nodes(), (childName) => {
        // Short-circuit if this child is a bridge node or it's not a terminal
        // node in the direction we're interested in.
        let childNodeInfo = coreGraph.node(childName);
        if (childNodeInfo.node.type === NodeType.BRIDGE) {
          return;
        }
        let isTerminal = inbound
          ? !coreGraph.predecessors(childName)?.length
          : !coreGraph.successors(childName)?.length;
        if (!isTerminal) {
          return;
        }
        // Find or create a bridge node in the container for all structural
        // metaedges. It would have been nice to skip this step and simply
        // set a metaedge between the terminal node and the container node, but
        // in that case, something about the graph upsets dagre.layout()'s
        // longestPath algorithm (was getting errors due to an undefined).
        let structuralNodeName = getBridgeNodeName(inbound, nodeName, 'STRUCTURAL_TARGET');
        let structuralRenderInfo = coreGraph.node(structuralNodeName);
        if (!structuralRenderInfo) {
          let bridgeNode: BridgeNode = {
            // Important Node properties.
            name: structuralNodeName,
            type: NodeType.BRIDGE,
            // Unimportant Node properties.
            isGroupNode: false,
            cardinality: 1,
            parentNode: null,
            include: InclusionType.UNSPECIFIED,
            // BridgeNode properties.
            inbound: inbound,
            inputData: {},
            outputData: {},
            suggestions: {},
            nodeAttributes: {},
          };
          structuralRenderInfo = new RenderNodeInfo(bridgeNode);
          structuralRenderInfo.structural = true;
          this.index[structuralNodeName] = structuralRenderInfo as any;
          coreGraph.setNode(structuralNodeName, structuralRenderInfo);
          bridgeContainerInfo.node.cardinality++;
          coreGraph.setParent(structuralNodeName, bridgeContainerName);
        }
        // Create the structural Metaedge and insert it.
        let structuralMetaedgeInfo = new RenderMetaedgeInfo(null);
        structuralMetaedgeInfo.structural = true;
        structuralMetaedgeInfo.weight--; // Reduce weight for dagre layout.
        inbound
          ? coreGraph.setEdge(structuralNodeName, childName, structuralMetaedgeInfo)
          : coreGraph.setEdge(childName, structuralNodeName, structuralMetaedgeInfo);
      });
    });
  }

  checkSubhierarchy(nodeName: string): boolean {
    return nodeName in this.hasSubhierarchy;
  }

  /**
   * Clones an op node and adds it to a metagraph. Does nothing if an op node
   * with the same new name has already been created within the metagraph. This
   * method is used when duplicating a library function to be injected within a
   * metanode representing a function call.
   * @param parentMetanode The parent metanode on which to add the new node.
   * @param node The op node to clone.
   * @param newPrefix The prefix string to use in lieu of the one that merely
   *     indicates that the metanode represents a function defined in the
   *     library. This prefix should reflect graph hierarchy.
   * @return The newly created op node (the clone of the original).
   */
  private cloneAndAddFunctionOpNode(
    parentMetanode: Metanode,
    libraryFunctionNodeName: string,
    node: OpNode,
    newPrefix: string,
  ): OpNode {
    const newName = node.name.replace(libraryFunctionNodeName, newPrefix);
    let newOpNode = parentMetanode.metagraph.node(newName) as any;
    if (newOpNode) {
      // This node had already been created and added to the graph.
      return newOpNode;
    }
    // Create a new op node.
    newOpNode = new OpNodeImpl({
      name: newName,
      input: [],
      op: node.op,
      input_data: _.cloneDeep(node.inputData),
      output_data: _.cloneDeep(node.outputData),
      stack_info: _.cloneDeep(node.stackData),
      suggestions: _.cloneDeep(node.suggestions),
      isLeaf: false,
      attr: _.cloneDeep(node.attr),
      node_type: NodeOpType.DEFAULT,
      matched_node_link: _.cloneDeep(node.matchedNodeLink),
    });
    // Update various properties.
    newOpNode.cardinality = node.cardinality;
    newOpNode.include = node.include;
    newOpNode.outputShapes = _.cloneDeep(node.outputShapes);
    // Update the inputs of the new node to reflect the new path.
    newOpNode.inputs = node.inputs.map((normalizedInput) => {
      const newNormalizedInput = _.clone(normalizedInput);
      newNormalizedInput.name = normalizedInput.name.replace(libraryFunctionNodeName, newPrefix);
      return newNormalizedInput;
    });
    // Add the new op node to the hierarchy and metagraph. Also add it to its
    // parent metanode.
    newOpNode.parentNode = parentMetanode;
    parentMetanode.metagraph.setNode(newOpNode.name, newOpNode);
    this.hierarchy.setNode(newOpNode.name, newOpNode);
    // Update embeddings.
    const updateEmbeddingOpNode = (embeddingNode): OpNode => {
      return this.cloneAndAddFunctionOpNode(parentMetanode, libraryFunctionNodeName, embeddingNode, newPrefix);
    };
    newOpNode.inEmbeddings = node.inEmbeddings.map(updateEmbeddingOpNode);
    newOpNode.outEmbeddings = node.outEmbeddings.map(updateEmbeddingOpNode);
    return newOpNode;
  }
}
/**
 * A class for rendering annotation object which contains label
 * about the node embedded as annotation, type of annotation and the location
 * of both the annotation's node and edge.
 *
 * Annotation objects include embedded constants, embedded summary, and
 * edge shortcuts.
 */
export class Annotation {
  node: Node;
  renderNodeInfo: RenderNodeInfo;
  renderMetaedgeInfo: RenderMetaedgeInfo | null;
  annotationType: AnnotationType;
  /**
   * Center position of annotation relative to the host
   * node's center x.
   */
  dx: number;
  /**
   * Center position of annotation relative to the host
   * node's center y.
   */
  dy: number;
  width: number;
  height: number;
  /**
   * The names of nodes on either side of this edge.
   */
  v: string;
  w: string;
  /**
   * A flag whether it is an in-annotation (if true) or
   * out-annotation  (if false).
   */
  isIn: boolean;
  /** Label horizontal offset from the end of the node shape */
  labelOffset: number;
  /**
   * Array of points for edges from the annotation to its host
   * node. Each point contains the point location, relative to
   * the host node's center.
   */
  points: {
    dx: number;
    dy: number;
  }[];
  /**
   * Creates a new Annotation.
   *
   * @param node The underlying node this annotation points to.
   * @param renderNodeInfo The render information for the underlying node
   *     this annotation points to. This can be null if the annotation
   *     denotes an embedding (constant, summary), in which case we
   *     use the node property.
   * @param renderMetaedgeInfo The render information for the edge associated
   *     with the annotation.
   * @param type The type of the annotation.
   * @param isIn True if it is an in-annotation. False if it is an
   *     out-annotation.
   */
  constructor(
    node: Node,
    renderNodeInfo: RenderNodeInfo,
    renderMetaedgeInfo: RenderMetaedgeInfo | null,
    type: AnnotationType,
    isIn: boolean,
  ) {
    this.node = node;
    this.renderNodeInfo = renderNodeInfo;
    this.renderMetaedgeInfo = renderMetaedgeInfo;
    this.annotationType = type;
    // Properties specified by layout
    this.dx = 0;
    this.dy = 0;
    this.width = 0;
    this.height = 0;
    // Properties needed for generating an ID for the edge's path element if
    // this annotation is associated with a metaedge.
    if (renderMetaedgeInfo?.metaedge) {
      this.v = renderMetaedgeInfo.metaedge.v ?? '';
      this.w = renderMetaedgeInfo.metaedge.w ?? '';
    }
    this.isIn = isIn;
    this.points = [];
  }
}
export enum AnnotationType {
  SHORTCUT = 0,
  CONSTANT = 1,
  SUMMARY = 2,
  ELLIPSIS = 3,
}
/**
 * Manages a list of annotations. Two will be used for each
 * RenderNodeInfo, one for in annotations and one for out annotations.
 */
export class AnnotationList {
  /**
   * List of visually drawable annotations, may include an ellipses annotation
   * if the number added exceeds the number specified by maxAnnotations.
   */
  list: Annotation[];
  /**
   * Set of nodes which have been added as annotations to this list, so we can
   * prevent duplicates.
   */
  nodeNames: {
    [nodeName: string]: boolean;
  };
  constructor() {
    this.list = [];
    this.nodeNames = {};
  }

  /**
   * Append an annotation to the list, or a stand-in ellipsis annotation instead
   * if this would make it too many.
   */
  push(annotation: Annotation): void {
    if (annotation.node.name in this.nodeNames) {
      return; // Skip duplicate annotation.
    }
    this.nodeNames[annotation.node.name] = true;
    if (this.list.length < PARAMS.maxAnnotations) {
      this.list.push(annotation);
      return;
    }
    let lastAnnotation = this.list[this.list.length - 1];
    if (lastAnnotation.annotationType === AnnotationType.ELLIPSIS) {
      let ellipsisNode = <EllipsisNode>lastAnnotation.node;
      ellipsisNode.setNumMoreNodes(++ellipsisNode.numMoreNodes);
      return;
    }
    let ellipsisNode = new tf_graph.EllipsisNodeImpl(1);
    this.list.push(
      new Annotation(ellipsisNode, new RenderNodeInfo(ellipsisNode), null, AnnotationType.ELLIPSIS, annotation.isIn),
    );
  }
}
/**
 * Contains rendering information about a node in the hierarchical graph.
 */
export class RenderNodeInfo {
  /** Reference to the original underlying Node from the hierarchical graph. */
  node: Node;
  /** Whether the node is expanded or not. */
  expanded: boolean;
  /**
   * List of rendering information about in-annotations like constants and
   * shortcuts to high-degree nodes.
   */
  inAnnotations: AnnotationList;
  /**
   * List of rendering information about out-annotations (e.g. summary nodes)
   */
  outAnnotations: AnnotationList;
  // --- Params specified by layout --- //
  /** Center x position */
  x: number;
  /** Center y position */
  y: number;
  /**
   * Total width of the node's shape, including in- and out-annotations. This
   * property is used by dagre to layout the graph.
   */
  width: number;
  /**
   * Total height of the node's shape, including in- and out-annotations. This
   * property is used by dagre to layout the graph.
   */
  height: number;
  /**
   * Size of the main box of the node, excluding in- and out-annotations. This
   * property is used to draw the rectangle/ellipse shape denoting the node.
   */
  coreBox: {
    width: number;
    height: number;
  };
  /** Width of the bounding box for all in-annotations. */
  inboxWidth: number;
  /** Width of the bounding box for all out-annotations. */
  outboxWidth: number;
  /**
   * Whether the node should be excluded from the scene.
   * This is only used when there are too many items in a series so we only
   * want to include top N ones.
   */
  excluded: boolean;
  // --- Params used in drawing the bridge paths --- //
  /**
   * All bridge nodes are meant to be invisible, but whereas most represent a
   * relationship from the underlying graph hierarchy, some exist solely for
   * layout reasons. Specifically, those bridge nodes which have only structural
   * rendering metaedges.
   */
  structural: boolean;
  // --- Params for the size of the node box --- //
  /** Label vertical offset from the center of node shape */
  labelOffset: number;
  /** Rectangle radius (for making rounded rectangle) */
  radius: number;
  // --- Params for expanded node --- //
  /** Label height for expanded node. */
  labelHeight: number;
  // Paddings between inner subscene and the border of the expanded node.
  paddingTop: number;
  paddingLeft: number;
  paddingRight: number;
  paddingBottom: number;
  /**
   * Whether a node is extracted as source-like (having high out-degree or
   * matching predefined in-extract pattern.)
   */
  isInExtract: boolean;
  /**
   * Whether a node is extracted as sink-like (having high in-degree or matching
   * predefined out-extract pattern.)
   */
  isOutExtract: boolean;
  /**
   * Whether a node represents a function template within the library, in which
   * case it should be rendered in a special scene group.
   */
  isLibraryFunction: boolean;
  /**
   * Whether this node is faded out. Used when displaying stats.
   */
  isFadedOut: boolean;
  /**
   * The name string used to label the node in the graph.
   */
  displayName: string;
  constructor(node: Node) {
    this.node = node;
    this.expanded = false;
    this.inAnnotations = new AnnotationList();
    this.outAnnotations = new AnnotationList();
    // Params specified by layout
    this.x = 0;
    this.y = 0;
    this.width = 0;
    this.height = 0;
    this.inboxWidth = 0;
    this.outboxWidth = 0;
    this.excluded = false;
    // Params for bridge paths.
    this.structural = false;
    // Params for node box.
    this.labelOffset = 0;
    this.radius = 0;
    // Params for expanded node
    this.labelHeight = 0;
    this.paddingTop = 0;
    this.paddingLeft = 0;
    this.paddingRight = 0;
    this.paddingBottom = 0;
    this.isInExtract = false;
    this.isOutExtract = false;
    this.coreBox = { width: 0, height: 0 };
    // By default, we don't fade nodes out. Default to false for safety.
    this.isFadedOut = false;
    // Only use the portion beyond the prefix as the display name.
    if (node.name.startsWith(BENCH_PREFIX) && node.parentNode.name === tf_graph.ROOT_NAME) {
      this.displayName = '标杆';
    } else {
      const nameList = node.name.split('.');
      if (nameList.length > 3) {
        const secondLastItem = nameList[nameList.length - 2];
        nameList.splice(nameList.length - 2, 1);
        nameList.splice(2, 0, secondLastItem);
        this.displayName = nameList.slice(1, nameList.length - 1).join('.');
      } else if (node.name.startsWith(BENCH_PREFIX) || node.name.startsWith(NPU_PREFIX)) {
        this.displayName = node.name.slice(4);
      } else {
        this.displayName = node.name;
      }
    }

    if (
      node.type === (NodeType.META || NodeType.MULTI_COLLECTION || NodeType.API_LIST) &&
      (node as Metanode).associatedFunction
    ) {
      // Function names are suffixed with a length-8 hexadecimal string
      // followed by an optional number. We remove that suffix because
      // the user did not generate that suffix. That suffix merely
      // serves to differentiate between functions with different
      // signatures but the same name otherwise.
      // Furthermore, we remove the prefix that merely ascertains this
      // node as a function definition. There is no reason for the user
      // to see that in the graph, as the node would already be within
      // the functions scene group.
      const match = this.displayName.match(nodeDisplayNameRegex);
      if (match) {
        // The display name had been successfully extracted. This is the most
        // common scenario.
        this.displayName = match[1];
      } else if (_.startsWith(this.displayName, tf_graph.FUNCTION_LIBRARY_NODE_PREFIX)) {
        // The string does not match the usual pattern for how functions are
        // named. Just use the entire second portion of the string as the name
        // if we can successfully remove the prefix.
        this.displayName = this.displayName.substring(tf_graph.FUNCTION_LIBRARY_NODE_PREFIX.length);
      }
    }
  }

  isInCore(): boolean {
    return !this.isInExtract && !this.isOutExtract && !this.isLibraryFunction;
  }
}
/**
 * Contains rendering information about a Metaedge from the underlying
 * hierarchical graph. It may be from either a metagraph or a bridgegraph.
 */
export class RenderMetaedgeInfo {
  /**
   * Reference to the original underlying Metaedge from the hierarchical graph,
   * if any. This will be null for the edges which connect OpNodes to their
   * embeddings, for example.
   */
  metaedge: Metaedge | null;
  /**
   * Reference to the adjoining RenderMetaedgeInfo from the parent's
   * coreGraph. This is used during layout to determine the point at which this
   * edge should touch the node's bounding box. This property will be null for
   * edges which terminate at a node on both ends (all non-bridge edges).
   */
  adjoiningMetaedge: RenderMetaedgeInfo | null;
  /**
   * Most of the time, a RenderMetaedgeInfo object represents a real
   * edge between nodes in the underlying graph structure. But sometimes, an
   * edge only exists for layout purposes. These structural edges are added
   * during buildSubhierarchy() to force dagre.layout() to put bridge nodes
   * at the ends of the flow.
   * @see buildSubhierarchy()
   */
  structural: boolean;
  /**
   * Weight of the edge, used by dagre when deciding how important an edge is.
   * Edges with higher weight are made shorter and straighter. The default
   * dagre uses is 1.
   */
  weight: number;
  /**
   * X and Y coordinate pairs of the points in the path of the edge.
   * @see tf_graph.node.subsceneAdjustPaths
   */
  points: Point[];
  /**
   * D3 selection of the group containing the path that displays this edge.
   */
  edgeGroup: d3.Selection<RenderMetaedgeInfo & any, any, any, any>;
  /** Id of the <marker> used as a start-marker for the edge path. */
  startMarkerId: string;
  /** Id of the <marker> used as an end-marker for the edge path. */
  endMarkerId: string;
  /**
   * Whether this edge is faded out. Used for fading out unused edges when
   * displaying run statistics.
   */
  isFadedOut: boolean;
  constructor(metaedge: Metaedge | null) {
    this.metaedge = metaedge;
    this.adjoiningMetaedge = null;
    this.structural = false;
    this.weight = 1;
    this.isFadedOut = false;
  }
}
function addInAnnotation(
  node: RenderNodeInfo,
  predecessor: Node,
  predecessorRenderInfo: RenderNodeInfo,
  edge: RenderMetaedgeInfo,
  type: AnnotationType,
): void {
  let annotation = new Annotation(predecessor, predecessorRenderInfo, edge, type, true);
  node.inAnnotations.push(annotation);
}
function addOutAnnotation(
  node: RenderNodeInfo,
  successor: Node,
  successorRenderInfo: RenderNodeInfo,
  edge: RenderMetaedgeInfo,
  type: AnnotationType,
): void {
  let annotation = new Annotation(successor, successorRenderInfo, edge, type, false);
  node.outAnnotations.push(annotation);
}
function setGraphDepth(graph: graphlib.Graph, depth: number): void {
  _.each(graph.nodes(), (nodeName) => {
    let child = graph.node(nodeName);
    child.expanded = depth > 1; // set all child of depth 1 to collapsed
    if (depth > 0) {
      switch (child.node.type) {
        case NodeType.META:
        case NodeType.MULTI_COLLECTION:
        case NodeType.API_LIST:
        case NodeType.SERIES:
          setGroupNodeDepth(<RenderGroupNodeInfo>child, depth - 1);
          break;
        default:
        // Do nothing for leaf
      }
    }
  });
}
export class RenderGroupNodeInfo extends RenderNodeInfo {
  override node: GroupNode;
  /**
   * The core graph is derived from the underlying node's metagraph, minus
   * the extracted source-like and sink-like nodes.
   */
  coreGraph: graphlib.Graph;
  /** Size of the bounding box for a metanode's isolated in-extract children. */
  inExtractBox: {
    width: number;
    height: number;
  };
  /**
   * Size of the bounding box for a metanode's isolated out-extract children.
   */
  outExtractBox: {
    width: number;
    height: number;
  };
  /** Size of the bounding box for the function library. */
  libraryFunctionsBox: {
    width: number;
    height: number;
  };
  /** Array of isolated in-extract nodes. */
  isolatedInExtract: RenderNodeInfo[];
  /** Array of isolated out-extract nodes. */
  isolatedOutExtract: RenderNodeInfo[];
  /** Array of nodes to show in the function library scene group. */
  libraryFunctionsExtract: RenderNodeInfo[];
  constructor(groupNode: GroupNode, graphOptions: tf_graph.LabeledGraphOptions) {
    super(groupNode);
    let metagraph = groupNode.metagraph;
    let gl = metagraph.graph() as any;
    this.coreGraph = createGraph<RenderNodeInfo, RenderMetaedgeInfo>(gl.name, GraphType.CORE, graphOptions);
    this.inExtractBox = { width: 0, height: 0 };
    this.outExtractBox = { width: 0, height: 0 };
    this.libraryFunctionsBox = { width: 0, height: 0 };
    this.isolatedInExtract = [];
    this.isolatedOutExtract = [];
    this.libraryFunctionsExtract = [];
  }
}
function setGroupNodeDepth(renderInfo: RenderGroupNodeInfo, depth: number): void {
  if (renderInfo.coreGraph) {
    setGraphDepth(renderInfo.coreGraph, depth);
  }
}
/**
 * Remove an edge from the graph and add annotations to both ends of the edge.
 *
 * @param The core graph.
 * @param v Source name.
 * @param w Sink name.
 */
function createShortcut(graph: graphlib.Graph, v: string, w: string): void {
  let src = graph.node(v) as any;
  let sink = graph.node(w) as any;
  let edge = graph.edge(v, w) as any;
  // If either of the nodes is explicitly included in the main graph and
  // both nodes are in the main graph then do not create the shortcut
  // and instead keep the real edge.
  const isInclude = src.node.include === InclusionType.INCLUDE || sink.node.include === InclusionType.INCLUDE;
  const isExclude = src.node.include === InclusionType.EXCLUDE || sink.node.include === InclusionType.EXCLUDE;
  if (isInclude && !isExclude) {
    return;
  }

  // Add each annotation.
  addOutAnnotation(src, sink.node, sink, edge, AnnotationType.SHORTCUT);
  addInAnnotation(sink, src.node, src, edge, AnnotationType.SHORTCUT);
  // Remove the edge from the core graph.
  graph.removeEdge(v, w);
}
/**
 * Remove edges from a node, and set its isOutExtract property to true,
 * and remove the node and move it to isolatedOutExtract.
 *
 * If detachAllEdgesForHighDegree or forceDetach is true, extract all of its
 * edges. Otherwise, only extract all in-edges.
 */
function makeOutExtract(renderNode: RenderGroupNodeInfo, n: string, forceDetach?: boolean): void {
  let graph = renderNode.coreGraph;
  let child = graph.node(n) as any;
  child.isOutExtract = true;
  _.each(graph.predecessors(n), (p, index) => {
    createShortcut(graph, p, n);
  });
  if (PARAMS.detachAllEdgesForHighDegree || forceDetach) {
    _.each(graph.successors(n), (s, index) => {
      createShortcut(graph, n, s);
    });
  }
  // Remove the node from the core graph if it no longer has neighbors.
  if (graph.neighbors(n)?.length === 0) {
    child.node.include = InclusionType.EXCLUDE;
    renderNode.isolatedOutExtract.push(child);
    graph.removeNode(n);
  }
}
/**
 * Remove edges from a node, set its isInExtract property to true,
 * and remove the node and move it to isolatedInExtract.
 *
 * If detachAllEdgesForHighDegree or forceDetach is true, extract all of its
 * edges. Otherwise, only remove all out-edges.
 */
export function makeInExtract(renderNode: RenderGroupNodeInfo, n: string, forceDetach?: boolean): void {
  let graph = renderNode.coreGraph;
  let child = graph.node(n) as any;
  child.isInExtract = true;
  _.each(graph.successors(n), (s, index) => {
    createShortcut(graph, n, s);
  });
  if (PARAMS.detachAllEdgesForHighDegree || forceDetach) {
    _.each(graph.predecessors(n), (p, index) => {
      createShortcut(graph, p, n);
    });
  }
  // Remove the node from the core graph if it no longer has neighbors.
  if (graph.neighbors(n)?.length === 0) {
    child.node.include = InclusionType.EXCLUDE;
    renderNode.isolatedInExtract.push(child);
    graph.removeNode(n);
  }
}
/**
 * Check whether the node's type is a member of the given list of types.
 *
 * @param node Node.
 * @param types List of type to match.
 */
function hasTypeIn(node: Node, types: string[]): boolean {
  if (node.type === NodeType.OP) {
    for (let i = 0; i < types.length; i++) {
      if ((<OpNode>node).op === types[i]) {
        return true;
      }
    }
  } else if (node.type === NodeType.META) {
    let rootOpNode = (<Metanode>node).getRootOp();
    if (rootOpNode) {
      for (let i = 0; i < types.length; i++) {
        if (rootOpNode.op === types[i]) {
          return true;
        }
      }
    }
  }
  return false;
}
/**
 * Given an integer, picks a hue that is far apart from other colors.
 * The formula for picking color that avoid collision is:
 *     hue = (color range * golden ratio * index) % color range
 */
export function mapIndexToHue(id: number): number {
  let GOLDEN_RATIO = 1.61803398875;
  // Hue of 0 is reserved for the gray nodes.
  let MIN_HUE = 1;
  let MAX_HUE = 359;
  let COLOR_RANGE = MAX_HUE - MIN_HUE;
  return MIN_HUE + ((COLOR_RANGE * GOLDEN_RATIO * id) % COLOR_RANGE);
}
/**
 * Remove edges and add to annotation instead.
 *
 * For root node, consider predefined types for source and sink.
 * We do not extract predefined type from non-root so that Variables and the
 * sgd node (op type = 'NoOp') do not get extract from inside own group.
 *
 * The order of extraction is important here as swapping the order can totally
 * screw up the graph layout.
 *
 * @param {Render.Node} renderNode Node to manipulate.
 *   nodes from the main graph. If false, only exclude predefined nodes.
 */
/**
 * Expands nodes in the graph until the desired node is visible.
 *
 * @param scene The scene polymer component.
 * @param renderHierarchy The render hierarchy.
 * @param tensorName The name of a tensor.
 * @return A string that is the name of the node representing the given tensor.
 *     Note that the original tensor name might differ from this returned node
 *     name. Specifically, for instance, the tensor name usually ends with an
 *     output slot index (such as :0), while the node name lacks that suffix.
 */
export function expandUntilNodeIsShown(scene, renderHierarchy: RenderGraphInfo, tensorName: string): string {
  const splitTensorName = tensorName.split('/');
  // Graph names do not take into account the output slot. Strip it.
  const lastNodeNameMatch = splitTensorName[splitTensorName.length - 1].match(/(?<name>.*)\w+/);
  if (lastNodeNameMatch?.length === 2) {
    splitTensorName[splitTensorName.length - 1] = lastNodeNameMatch?.[1];
  }
  let nodeName = splitTensorName[0];
  let renderNode = renderHierarchy.getRenderNodeByName(nodeName);
  for (let i = 1; i < splitTensorName.length; i++) {
    // Op nodes are not expandable.
    if (renderNode.node.type === tf_graph.NodeType.OP) {
      break;
    }
    renderHierarchy.buildSubhierarchy(nodeName);
    renderNode.expanded = true;
    scene.setNodeExpanded(renderNode);
    nodeName += `/${splitTensorName[i]}`;
    renderNode = renderHierarchy.getRenderNodeByName(nodeName);
  }
  return renderNode.node.name;
}
