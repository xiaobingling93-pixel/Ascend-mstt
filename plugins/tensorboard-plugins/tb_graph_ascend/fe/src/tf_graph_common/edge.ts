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
import * as d3 from 'd3';
import { graphlib } from 'dagre';
import * as _ from 'lodash';
import * as tf_graph_common from './common';
import { Class, MAX_EDGE_WIDTH, MIN_EDGE_WIDTH } from './common';
import { BaseEdge, EDGE_KEY_DELIM, Metaedge, OpNode, TensorShape } from './graph';
import * as render from './render';
import { EdgeData } from './render';
import { TfGraphScene } from './tf-graph-scene';
import { safeJSONParse } from '../utils';

/** Delimiter between dimensions when showing sizes of tensors. */
const TENSOR_SHAPE_DELIM = '\u00D7';

let arrowheadMap = d3
  .scaleQuantize<string>()
  .domain([MIN_EDGE_WIDTH, MAX_EDGE_WIDTH])
  .range(['small', 'medium', 'large', 'xlarge']);
/** Minimum stroke width to put edge labels in the middle of edges */
const CENTER_EDGE_LABEL_MIN_STROKE_WIDTH = 2.5;

/**
 * Function run when an edge is selected.
 */
export interface EdgeSelectionCallback {
  (edgeData: EdgeData): void;
}
export function getEdgeKey(edgeObj: EdgeData): string {
  return `${edgeObj.v}${EDGE_KEY_DELIM}${edgeObj.w}${EDGE_KEY_DELIM}${edgeObj.id}`;
}
/**
 * Select or Create a 'g.edges' group to a given sceneGroup
 * and builds a number of 'g.edge' groups inside the group.
 *
 * Structure Pattern:
 *
 * <g class='edges'>
 *   <g class='edge'>
 *     <path class='edgeline'/>
 *   </g>
 *   ...
 * </g>
 *
 *
 * @param sceneGroup container
 * @param graph
 * @param sceneElement <tf-graph-scene> polymer element.
 * @return selection of the created nodeGroups
 */
export function buildGroup(sceneGroup, graph: graphlib.Graph, sceneElement: TfGraphScene): any {
  const sceneComponent = sceneElement as any;
  let edges: EdgeData[] = [];
  edges = _.reduce(
    graph.edges(),
    (edgesAcc, edgeObj) => {
      let edgeLabel = graph.edge(edgeObj);
      edgesAcc.push({
        v: edgeObj.v,
        w: edgeObj.w,
        id: edgeObj.name,
        label: edgeLabel,
      });
      return edges;
    },
    edges,
  );
  let container = tf_graph_common.selectOrCreateChild(sceneGroup, 'g', Class.Edge.CONTAINER);
  // Select all children and join with data.
  // (Note that all children of g.edges are g.edge)
  let edgeGroups = (container as any)
    .selectAll(function () {
      return this.childNodes;
    })
    .data(edges, getEdgeKey);
  // Make edges a group to support rendering multiple lines for metaedge
  edgeGroups
    .enter()
    .append('g')
    .attr('class', Class.Edge.GROUP)
    .attr('data-edge', getEdgeKey)
    .each(function (d: EdgeData) {
      let edgeGroup = d3.select(this);
      d.label.edgeGroup = edgeGroup;
      // index node group for quick highlighting
      sceneComponent._edgeGroupIndex[getEdgeKey(d)] = edgeGroup;
    })
    .merge(edgeGroups)
    .each(function () {
      position(sceneElement, this);
    })
    .each(function (d) {
      stylize(d3.select(this), d, sceneComponent);
    });
  edgeGroups
    .exit()
    .each((d) => {
      delete sceneComponent._edgeGroupIndex[getEdgeKey(d)];
    })
    .remove();
  return edgeGroups;
}
/**
 * Returns the label for the given base edge.
 * The label is the shape of the underlying tensor.
 */
export function getLabelForBaseEdge(baseEdge: BaseEdge, renderInfo: render.RenderGraphInfo): string {
  const outTensorKey = baseEdge.outputTensorKey;
  let shape: TensorShape = [];
  if (outTensorKey?.startsWith('[')) {
    shape = safeJSONParse(outTensorKey) as TensorShape;
  } else {
    let node = <OpNode>renderInfo.getNodeByName(baseEdge.v ?? '');
    if (node.outputShapes == null || _.isEmpty(node.outputShapes)) {
      return '';
    }
    shape = node.outputShapes[baseEdge.outputTensorKey];
  }
  if (shape == null) {
    return '';
  }
  if (shape.length === 0) {
    return 'scalar';
  }
  return shape
    .map((size) => {
      return size === -1 ? '?' : size;
    })
    .join(TENSOR_SHAPE_DELIM);
}

/**
 * Creates the label for the given metaedge. If the metaedge consists
 * of only 1 tensor, and it's shape is known, the label will contain that
 * shape. Otherwise, the label will say the number of tensors in the metaedge.
 */
export function getLabelForEdge(metaedge: Metaedge, renderInfo: render.RenderGraphInfo): string {
  // Compute the label based on either tensor count or size.
  let isMultiEdge = metaedge.baseEdgeList.length > 1;
  return isMultiEdge
    ? `${metaedge.baseEdgeList.length}tensors`
    : getLabelForBaseEdge(metaedge.baseEdgeList[0], renderInfo);
}

/**
 * For a given d3 selection and data object, create a path to represent the
 * edge described in d.label.
 *
 * If d.label is defined, it will be a RenderMetaedgeInfo instance. It
 * will sometimes be undefined, for example for some Annotation edges for which
 * there is no underlying Metaedge in the hierarchical graph.
 */
export function appendEdge(
  edgeGroup,
  d: EdgeData,
  sceneElement: {
    renderHierarchy: render.RenderGraphInfo;
    handleEdgeSelected?: () => void;
  },
  edgeClass?: string,
): void {
  let edgeClassNew = edgeClass ?? Class.Edge.LINE; // set default type
  if (d.label?.structural) {
    edgeClassNew += ` ${Class.Edge.STRUCTURAL}`;
  }
  if (d.label?.metaedge?.numRefEdges) {
    edgeClassNew += ` ${Class.Edge.REFERENCE_EDGE}`;
  }
  if (sceneElement.handleEdgeSelected) {
    // The user has opted to make edges selectable.
    edgeClassNew += ` ${Class.Edge.SELECTABLE}`;
  }
  // Give the path a unique id, which will be used to link
  // the textPath (edge label) to this path.
  let pathId = `path_${getEdgeKey(d)}`;
  let strokeWidth;
  // Encode tensor size within edge thickness.
  let size = 1;
  if (d.label?.metaedge) {
    // There is an underlying Metaedge.
    size = d.label.metaedge.totalSize;
  }
  strokeWidth = sceneElement.renderHierarchy.edgeWidthSizedBasedScale(size);
  let path = edgeGroup
    .append('path')
    .attr('id', pathId)
    .attr('class', edgeClassNew)
    .style('stroke-width', `${strokeWidth}px`);
  // Check if there is a reference edge and add an arrowhead of the right size.
  if (d.label?.metaedge) {
    if (d.label.metaedge.numRefEdges) {
      // We have a reference edge.
      const markerId = `reference-arrowhead-${arrowheadMap(strokeWidth)}`;
      path.style('marker-start', `url(#${markerId})`);
      d.label.startMarkerId = markerId;
    } else {
      // We have a dataflow edge.
      const markerId = `dataflow-arrowhead-${arrowheadMap(strokeWidth)}`;
      path.style('marker-end', `url(#${markerId})`);
      d.label.endMarkerId = markerId;
    }
  }
  if (d.label == null || d.label.metaedge == null) {
    // There is no associated metaedge, thus no text.
    // This happens for annotation edges.
    return;
  }
  let labelForEdge = getLabelForEdge(d.label.metaedge, sceneElement.renderHierarchy);
  if (labelForEdge == null) {
    // We have no information to show on this edge.
    return;
  }
  edgeGroup
    .append('text')
    .append('textPath')
    .attr('xlink:href', `#${pathId}`)
    .attr('startOffset', '50%')
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'central')
    .text(labelForEdge);
}

export const interpolate: d3.Line<{
  x: number;
  y: number;
}> = d3
  .line<{
    x: number;
    y: number;
  }>()
  .curve(d3.curveBasis)
  .x((d) => {
    return d.x;
  })
  .y((d) => {
    return d.y;
  });

function position(component: HTMLElement, edgeGroup: HTMLElement): void {
  d3.select(edgeGroup).select(`path.${Class.Edge.LINE}`).transition();
}

/**
 * For a given d3 selection and data object, mark the edge as a control
 * dependency if it contains only control edges.
 *
 * d's label property will be a RenderMetaedgeInfo object.
 */
export function stylize(edgeGroup, d: EdgeData, sceneElement: TfGraphScene): void {
  edgeGroup.classed('faded', d.label.isFadedOut);
  let metaedge = d.label.metaedge;
  edgeGroup.select(`path.${Class.Edge.LINE}`).classed('control-dep', metaedge && !metaedge.numRegularEdges);
}
