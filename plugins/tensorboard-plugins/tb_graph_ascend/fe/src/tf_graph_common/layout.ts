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
import * as d3 from 'd3';
import * as dagre from 'dagre';
import { graphlib } from 'dagre';
import * as _ from 'lodash';
import { NodeType } from './graph';
import * as render from './render';

export const PARAMS = {
  animation: {
    /** Default duration for graph animations in ms. */
    duration: 250,
  },
  graph: {
    /** Graph parameter for metanode. */
    meta: {
      /**
       * Dagre's nodesep param - number of pixels that
       * separate nodes horizontally in the layout.
       *
       */
      nodeSep: 5,
      /**
       * Dagre's edgesep param - number of pixels that separate
       * edges horizontally in the layout.
       */
      edgeSep: 5,
    },
    /**
     * Padding is used to correctly position the graph SVG inside of its parent
     * element. The padding amounts are applied using an SVG transform of X and
     * Y coordinates.
     */
    padding: { paddingTop: 40, paddingLeft: 20 },
  },
  subscene: {
    meta: {
      paddingTop: 5,
      paddingBottom: 5,
      paddingLeft: 8,
      paddingRight: 8,
      /**
       * Used to leave room for the label on top of the highest node in
       * the core graph.
       */
      labelHeight: 20,
      /** X-space between each extracted node and the core graph. */
      extractXOffset: 15,
      /** Y-space between each extracted node. */
      extractYOffset: 20,
    },
  },
  nodeSize: {
    /** Size of meta nodes. */
    meta: {
      radius: 5,
      width: 60,
      maxLabelWidth: 200,
      /** A scale for the node's height based on number of nodes inside */
      // Hack - set this as an any type to avoid issues in exporting a type
      // from an external module.
      height: (d3 as any).scaleLinear().domain([1, 200]).range([15, 60]).clamp(true),
      /** The radius of the circle denoting the expand button. */
      expandButtonRadius: 3,
    },
    multi_collection: {
      radius: 5,
      width: 60,
      maxLabelWidth: 200,
      /** A scale for the node's height based on number of nodes inside */
      // Hack - set this as an any type to avoid issues in exporting a type
      // from an external module.
      height: (d3 as any).scaleLinear().domain([1, 200]).range([15, 60]).clamp(true),
      /** The radius of the circle denoting the expand button. */
      expandButtonRadius: 3,
    },
    api_list: {
      radius: 5,
      width: 60,
      maxLabelWidth: 200,
      /** A scale for the node's height based on number of nodes inside */
      // Hack - set this as an any type to avoid issues in exporting a type
      // from an external module.
      height: (d3 as any).scaleLinear().domain([1, 200]).range([15, 60]).clamp(true),
      /** The radius of the circle denoting the expand button. */
      expandButtonRadius: 3,
    },
    /** Size of op nodes. */
    op: {
      width: 30,
      height: 12,
      radius: 6,
      labelOffset: -12,
      maxLabelWidth: 40,
    },
  },
  shortcutSize: {
    /** Size of shortcuts for op nodes */
    op: { width: 10, height: 4 },
    /** Size of shortcuts for meta nodes */
    meta: { width: 12, height: 4, radius: 1 },
    /** Size of shortcuts for multi_collection nodes */
    multi_collection: { width: 12, height: 4, radius: 1 },
    /** Size of shortcuts for api_list nodes */
    api_list: { width: 12, height: 4, radius: 1 },
  },
  annotations: {
    /** Maximum possible width of the bounding box for in annotations */
    inboxWidth: 50,
    /** Maximum possible width of the bounding box for out annotations */
    outboxWidth: 50,
    /** X-space between the shape and each annotation-node. */
    xOffset: 10,
    /** Y-space between each annotation-node. */
    yOffset: 3,
    /** X-space between each annotation-node and its label. */
    labelOffset: 2,
    /** Defines the max width for annotation label */
    maxLabelWidth: 40,
  },
  constant: { size: { width: 4, height: 4 } },
  minimap: {
    /** The maximum width/height the minimap can have. */
    size: 150,
  },
};
/**
 * The minimum width we confer upon the auxiliary nodes section if functions
 * also appear. Without enforcing this minimum, metanodes in the function
 * library section could jut into the auxiliary nodes section because the
 * title "Auxiliary Nodes" is longer than the width of the auxiliary nodes
 * section itself.
 */
export const MIN_AUX_WIDTH = 140;
/**
 * Keep this number as the same as 'maxMetanodeLabelLength' in 'tf-graph-scene'
 */
export const MAX_TEXT_LENGTH = 50;
/**
 * 6 pixels per character.
 */
export const CHARACTER_WIDTH = 6;
/** Calculate layout for a scene of a group node. */
export function layoutScene(renderNodeInfo: render.RenderGroupNodeInfo): void {
  // Update layout, size, and annotations of its children nodes and edges.
  if (renderNodeInfo.node.isGroupNode) {
    layoutChildren(renderNodeInfo);
  }
  // Update position of its children nodes and edges
  if (renderNodeInfo.node.type === NodeType.META) {
    layoutMetanode(renderNodeInfo, 10);
  } else if (renderNodeInfo.node.type === NodeType.MULTI_COLLECTION) {
    layoutMetanode(renderNodeInfo, 10);
  } else if (renderNodeInfo.node.type === NodeType.API_LIST) {
    layoutMetanode(renderNodeInfo, 32);
  } else {
  }
}
/**
 * Updates the total width of an unexpanded node which includes the size of its
 * in and out annotations.
 */
function updateTotalWidthOfNode(renderInfo: render.RenderNodeInfo): void {
  // Assign the width of the core box (the main shape of the node).
  renderInfo.coreBox.width = renderInfo.width;
  renderInfo.coreBox.height = renderInfo.height;
  let labelLength = renderInfo.displayName.length;
  // Compute the total width of the node.
  if (renderInfo.node.type === NodeType.OP) {
    renderInfo.width = PARAMS.nodeSize.op.maxLabelWidth;
  } else {
    renderInfo.width = Math.max(
      renderInfo.coreBox.width,
      Math.min(labelLength * CHARACTER_WIDTH, PARAMS.nodeSize.meta.maxLabelWidth),
    );
  }
}
/**
 * Update layout, size, and annotations of its children nodes and edges.
 */
function layoutChildren(renderNodeInfo: render.RenderGroupNodeInfo): void {
  let children = renderNodeInfo.coreGraph
    .nodes()
    .map((n) => {
      return renderNodeInfo.coreGraph.node(n);
    })
  _.each(children, (childNodeInfo) => {
    // Set size of each child
    switch (childNodeInfo.node.type) {
      case NodeType.OP:
        _.extend(childNodeInfo, PARAMS.nodeSize.op);
        break;
      case NodeType.META:
      case NodeType.MULTI_COLLECTION:
      case NodeType.API_LIST:
        if (!childNodeInfo.expanded) {  
          // Set fixed width and scalable height based on cardinality
          _.extend(childNodeInfo, PARAMS.nodeSize.meta);
          childNodeInfo.height = PARAMS.nodeSize.meta.height(childNodeInfo.node.cardinality);
          childNodeInfo.width = Math.max(
            childNodeInfo.width,
            Math.min(childNodeInfo.displayName.length, MAX_TEXT_LENGTH) * CHARACTER_WIDTH,
          );
        } else {
          let childGroupNodeInfo = <render.RenderGroupNodeInfo>childNodeInfo;
          layoutScene(childGroupNodeInfo); // Recursively layout its subscene.
        }
        break;
      default:
        throw Error(`Unrecognized node type: ${childNodeInfo.node.type}`);
    }
    // Compute total width of un-expanded nodes. Width of expanded nodes
    // has already been computed.
    if (!childNodeInfo.expanded) {
      updateTotalWidthOfNode(childNodeInfo);
    }
  });
}
/**
 * Calculate layout for a graph using dagre
 * @param graph the graph to be laid out
 * @param params layout parameters
 * @return width and height of the core graph
 */
function dagreLayout(graph: graphlib.Graph, params): { height: number; width: number } {
  _.extend(graph.graph(), {
    nodesep: params.nodeSep,
    ranksep: params.rankSep,
    edgesep: params.edgeSep,
  });
  dagre.layout(graph);
  // Calculate the true bounding box of the graph by iterating over nodes and
  // edges rather than accepting dagre's word for it. In particular, we should
  // ignore the extra-wide bridge nodes and bridge edges, and allow for
  // annotation boxes and labels.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  _.each(graph.nodes(), (nodeName) => {
    let nodeInfo = graph.node(nodeName);
    let w = 0.5 * nodeInfo.width;
    let x1 = nodeInfo.x - w;
    let x2 = nodeInfo.x + w;
    minX = x1 < minX ? x1 : minX;
    maxX = x2 > maxX ? x2 : maxX;
    let h = 0.5 * nodeInfo.height;
    let y1 = nodeInfo.y - h;
    let y2 = nodeInfo.y + h;
    minY = y1 < minY ? y1 : minY;
    maxY = y2 > maxY ? y2 : maxY;
  });

  _.each(graph.nodes(), (nodeName) => {
    let nodeInfo = graph.node(nodeName);
    nodeInfo.x -= minX;
    nodeInfo.y -= minY;
  });
  return {
    width: maxX - minX,
    height: maxY - minY,
  };
}
/** Layout a metanode. Only called for an expanded node. */
function layoutMetanode(renderNodeInfo: render.RenderGroupNodeInfo, rankSep: number): void {
  // First, copy params specific to meta nodes onto this render info object.
  let params = PARAMS.subscene.meta;
  _.extend(renderNodeInfo, params);
  // Invoke dagre.layout() on the core graph and record the bounding box
  // dimensions.
  _.extend(renderNodeInfo.coreBox, dagreLayout(renderNodeInfo.coreGraph, { ...PARAMS.graph.meta, rankSep }));
  // Compute the total padding between the core graph, in-extract and
  // out-extract boxes.
  let numParts = 0;
  if (renderNodeInfo.coreGraph.nodeCount() > 0) {
    numParts++;
  }
  let offset = PARAMS.subscene.meta.extractXOffset;
  let padding = numParts <= 1 ? 0 : numParts * offset;
  renderNodeInfo.coreBox.width += padding + padding;
  renderNodeInfo.coreBox.height = params.labelHeight + renderNodeInfo.coreBox.height,
  // Determine the whole metanode's width (from left to right).
  renderNodeInfo.width =
    Math.max(renderNodeInfo.displayName.length * CHARACTER_WIDTH, renderNodeInfo.coreBox.width) +
    params.paddingLeft +
    params.paddingRight;
  // Determine the whole metanode's height (from top to bottom).
  renderNodeInfo.height = renderNodeInfo.paddingTop + renderNodeInfo.coreBox.height + renderNodeInfo.paddingBottom;
}

/**
 * Determines the center position of the node's shape. The position depends
 * on if the node has in and out-annotations.
 */
export function computeCXPositionOfNodeShape(renderInfo: render.RenderNodeInfo): number {
  if (renderInfo.expanded) {
    return renderInfo.x;
  }
  return renderInfo.x - (renderInfo.width / 2) + (renderInfo.coreBox.width / 2);
}
