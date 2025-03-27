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
import { Class as _Class, selectChild as _selectChild } from './common';
import { NodeType } from './graph';
import * as layout from './layout';
import * as render from './render';

export const selectChild = _selectChild;
export const Class = _Class;

/**
 * The dimensions of the minimap including padding and margin.
 */
const MINIMAP_BOX_WIDTH = 320;
const MINIMAP_BOX_HEIGHT = 150;
/**
 * Helper method for fitting the graph in the svg view.
 *
 * @param svg The main svg.
 * @param zoomG The svg group used for panning and zooming.
 * @param d3zoom The zoom behavior.
 * @param callback Called when the fitting is done.
 */
export function fit(svg, zoomG, d3zoom, callback): void {
  let svgRect = svg.getBoundingClientRect();
  let sceneSize: DOMRect | null = null;
  try {
    sceneSize = zoomG.getBBox();
    if (sceneSize?.width === 0) {
      // There is no scene anymore. We have been detached from the dom.
      return;
    }
  } catch (e) {
    // Firefox produced NS_ERROR_FAILURE if we have been
    // detached from the dom.
    return;
  }
  let scale = 0.9 * Math.min(svgRect.width / (sceneSize?.width ?? 1), svgRect.height / (sceneSize?.height ?? 1), 2);
  let params = layout.PARAMS.graph;
  const transform = d3.zoomIdentity.scale(scale).translate(params.padding.paddingLeft, params.padding.paddingTop);
  d3.select(svg)
    .transition()
    .duration(500)
    .call(d3zoom.transform, transform)
    .on('end.fitted', () => {
      // Remove the listener for the zoomend event,
      // so we don't get called at the end of regular zoom events,
      // just those that fit the graph to screen.
      d3zoom.on('end.fitted', null);
      callback();
    });
}
/**
 * Helper method for panning the graph to center on the provided node,
 * if the node is currently off-screen.
 *
 * @param nodeName The node to center the graph on
 * @param svg The root SVG element for the graph
 * @param zoomG The svg group used for panning and zooming.
 * @param d3zoom The zoom behavior.
 * @return True if the graph had to be panned to display the
 *            provided node.
 */
export function panToNode(nodeName: string, svg, zoomG, d3zoom): boolean {
  const node = <SVGAElement>d3.select(svg).select(`[data-name="${nodeName}"]`).node();
  if (!node) {
    console.warn(`panToNode failed for node name "${nodeName}"`);
    return false;
  }
  // Check if the selected node is off-screen in either
  // X or Y dimension in either direction.
  let nodeBox = node.getBBox();
  let nodeCtm = node.getScreenCTM();
  let pointTL = svg.createSVGPoint();
  let pointBR = svg.createSVGPoint();
  pointTL.x = nodeBox.x;
  pointTL.y = nodeBox.y;
  pointBR.x = nodeBox.x + nodeBox.width;
  pointBR.y = nodeBox.y + nodeBox.height;
  pointTL = pointTL.matrixTransform(nodeCtm);
  pointBR = pointBR.matrixTransform(nodeCtm);
  let isOutsideOfBounds = (start, end, lowerBound, upperBound): boolean => {
    // Return if even a part of the interval is out of bounds.
    return !(start > lowerBound && end < upperBound);
  };
  let svgRect = svg.getBoundingClientRect();
  // Subtract to make sure that the node is not hidden behind the minimap.
  const horizontalBound = svgRect.left + svgRect.width - MINIMAP_BOX_WIDTH;
  const verticalBound = svgRect.top + svgRect.height - MINIMAP_BOX_HEIGHT;
  if (
    isOutsideOfBounds(pointTL.x, pointBR.x, svgRect.left, horizontalBound) ||
    isOutsideOfBounds(pointTL.y, pointBR.y, svgRect.top, verticalBound)
  ) {
    // Determine the amount to translate the graph in both X and Y dimensions in
    // order to center the selected node. This takes into account the position
    // of the node, the size of the svg scene, the amount the scene has been
    // scaled by through zooming, and any previous transforms already performed
    // by this logic.
    let centerX = (pointTL.x + pointBR.x) / 2;
    let centerY = (pointTL.y + pointBR.y) / 2;
    let dx = svgRect.left + (svgRect.width / 2) - centerX;
    let dy = svgRect.top + (svgRect.height / 2) - centerY;

    // We translate by this amount. We divide the X and Y translations by the
    // scale to undo how translateBy scales the translations (in d3 v4).
    const svgTransform = d3.zoomTransform(svg);
    d3.select(svg)
      .transition()
      .duration(500)
      .call(d3zoom.translateBy, dx / svgTransform.k, dy / svgTransform.k);
    return true;
  }
  return false;
}
/**
 * Given a scene's svg group, set  g.in-extract, g.coreGraph, g.out-extract svg
 * groups' position relative to the scene.
 *
 * @param sceneGroup
 * @param renderNode render node of a metanode or series node.
 */
export function position(sceneGroup, renderNode: render.RenderGroupNodeInfo): void {
  // Translate scenes down by the label height so that when showing graphs in
  // expanded metanodes, the graphs are below the labels.  Do not shift them
  // down for series nodes as series nodes don't have labels inside of their
  // bounding boxes.
  let yTranslate = layout.PARAMS.subscene.meta.labelHeight;
  // core
  translate(selectChild(sceneGroup, 'g', Class.Scene.CORE), 0, yTranslate);
  // in-extract
  let hasInExtract = renderNode.isolatedInExtract.length > 0;
  let hasOutExtract = renderNode.isolatedOutExtract.length > 0;
  let hasLibraryFunctions = renderNode.libraryFunctionsExtract.length > 0;
  let offset = layout.PARAMS.subscene.meta.extractXOffset;
  let auxWidth = 0;
  if (hasInExtract) {
    auxWidth += renderNode.outExtractBox.width;
  }
  if (hasOutExtract) {
    auxWidth += renderNode.outExtractBox.width;
  }
  if (hasInExtract) {
    let inExtractX = renderNode.coreBox.width;
    if (auxWidth < layout.MIN_AUX_WIDTH) {
      inExtractX = inExtractX - layout.MIN_AUX_WIDTH + (renderNode.inExtractBox.width / 2);
    } else {
      inExtractX =
        inExtractX - (renderNode.inExtractBox.width / 2) - renderNode.outExtractBox.width - (hasOutExtract ? offset : 0);
    }
    inExtractX = inExtractX - renderNode.libraryFunctionsBox.width - (hasLibraryFunctions ? offset : 0);
    translate(selectChild(sceneGroup, 'g', Class.Scene.INEXTRACT), inExtractX, yTranslate);
  }
  // out-extract
  if (hasOutExtract) {
    let outExtractX = renderNode.coreBox.width;
    if (auxWidth < layout.MIN_AUX_WIDTH) {
      outExtractX = outExtractX - layout.MIN_AUX_WIDTH + (renderNode.outExtractBox.width / 2);
    } else {
      outExtractX -= renderNode.outExtractBox.width / 2;
    }
    outExtractX = outExtractX - renderNode.libraryFunctionsBox.width - (hasLibraryFunctions ? offset : 0);
    translate(selectChild(sceneGroup, 'g', Class.Scene.OUTEXTRACT), outExtractX, yTranslate);
  }
  if (hasLibraryFunctions) {
    let libraryFunctionsExtractX = renderNode.coreBox.width - (renderNode.libraryFunctionsBox.width / 2);
    translate(selectChild(sceneGroup, 'g', Class.Scene.FUNCTION_LIBRARY), libraryFunctionsExtractX, yTranslate);
  }
}
/** Adds a click listener to a group that fires a graph-select event */
export function addGraphClickListener(graphGroup, sceneElement): void {
  d3.select(graphGroup).on('click', () => {
    sceneElement.fire('graph-select');
  });
}
/** Helper for adding transform: translate(x0, y0) */
export function translate(selection, x0: number, y0: number): void {
  // If it is already placed on the screen, make it a transition.
  let selectionTemp = selection;
  if (selection.attr('transform') != null) {
    selectionTemp = selection.transition('position');
  }
  selectionTemp.attr('transform', `translate(${x0},${y0})`);
}
/**
 * Helper for setting position of a svg rect
 * @param rect A d3 selection of rect(s) to set position of.
 * @param cx Center x.
 * @param cy Center x.
 * @param width Width to set.
 * @param height Height to set.
 */
export function positionRect(rect, cx: number, cy: number, width: number, height: number): void {
  rect
    .transition()
    .attr('x', cx - (width / 2))
    .attr('y', cy - (height / 2))
    .attr('width', width)
    .attr('height', height);
}

/**
 * Helper for setting position of a svg expand/collapse button
 * @param button container group
 * @param renderNode the render node of the group node to position
 *        the button on.
 */
export function positionButton(button, renderNode: render.RenderNodeInfo): void {
  let cx = layout.computeCXPositionOfNodeShape(renderNode);
  // Position the button in the top-right corner of the group node,
  // with space given the draw the button inside of the corner.
  let width = renderNode.expanded ? renderNode.width : renderNode.coreBox.width;
  let height = renderNode.expanded ? renderNode.height : renderNode.coreBox.height;
  let x = cx + (width / 2) - 6;
  let y = renderNode.y - (height / 2) + 6;
  let translateStr = `translate(${x},${y})`;
  button.selectAll('path').transition().attr('transform', translateStr);
  button.select('circle').transition().attr({ cx: x, cy: y, r: layout.PARAMS.nodeSize.meta.expandButtonRadius });
}
/**
 * Helper for setting position of a svg ellipse
 * @param ellipse ellipse to set position of.
 * @param cx Center x.
 * @param cy Center x.
 * @param width Width to set.
 * @param height Height to set.
 */
export function positionEllipse(ellipse, cx: number, cy: number, width: number, height: number): void {
  ellipse
    .transition()
    .attr('cx', cx)
    .attr('cy', cy)
    .attr('rx', width / 2)
    .attr('ry', height / 2);
}
