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
 * @fileoverview Common interfaces for the tensorflow graph visualizer.
 */
import * as d3 from 'd3';

export interface ProgressTracker {
  updateProgress: (incrementValue: number) => void;
  setMessage: (msg: string) => void;
  reportError: (msg: string, err: Error) => void;
}
// Note that tf-graph-control depends on the value of the enum.
// Polymer does not let one use JS variable as a prop.
export enum SelectionType {
  OP_GRAPH = 'op_graph',
  CONCEPTUAL_GRAPH = 'conceptual_graph',
  PROFILE = 'profile',
}

/** Enums element class of objects in the scene */
export const Class = {
  Node: {
    // <g> element that contains nodes.
    CONTAINER: 'nodes',
    // <g> element that contains detail about a node.
    GROUP: 'node',
    // <g> element that contains visual elements (like rect, ellipse).
    SHAPE: 'nodeshape',
    OUTER: 'outer',
    // <*> element(s) under SHAPE that should receive color updates.
    COLOR_TARGET: 'nodecolortarget',
    // <text> element showing the node's label.
    LABEL: 'nodelabel',
    // <g> element that contains all visuals for the expand/collapse
    // button for expandable group nodes.
    BUTTON_CONTAINER: 'buttoncontainer',
    // <circle> element that surrounds expand/collapse buttons.
    BUTTON_CIRCLE: 'buttoncircle',
    // <path> element of the expand button.
    EXPAND_BUTTON: 'expandbutton',
    // <path> element of the collapse button.
    COLLAPSE_BUTTON: 'collapsebutton',
  },
  Edge: {
    CONTAINER: 'edges',
    GROUP: 'edge',
    LINE: 'edgeline',
    REFERENCE_EDGE: 'referenceedge',
    REF_LINE: 'refline',
    SELECTABLE: 'selectableedge',
    SELECTED: 'selectededge',
    STRUCTURAL: 'structural',
    HIGHLIGHTED: 'highlighted',
  },
  Annotation: {
    OUTBOX: 'out-annotations',
    INBOX: 'in-annotations',
    GROUP: 'annotation',
    NODE: 'annotation-node',
    EDGE: 'annotation-edge',
    CONTROL_EDGE: 'annotation-control-edge',
    LABEL: 'annotation-label',
    ELLIPSIS: 'annotation-ellipsis',
  },
  Scene: {
    GROUP: 'scene',
    CORE: 'core',
    FUNCTION_LIBRARY: 'function-library',
    INEXTRACT: 'in-extract',
    OUTEXTRACT: 'out-extract',
  },
  Subscene: { GROUP: 'subscene' },
  OPNODE: 'op',
  METANODE: 'meta',
  SERIESNODE: 'series',
  BRIDGENODE: 'bridge',
  ELLIPSISNODE: 'ellipsis',
  API_LIST: 'api_list',
  MULTI_COLLECTION: 'multi_collection',
};

// Please keep this in sync with tf-graph-scene.html.ts.
export const FontSizeInPx: Record<string, Record<string, number>> = {
  Edge: {
    LABEL: 3.5,
  },
  Annotation: {
    LABEL: 5,
  },
  Node: {
    EXPANDED_LABEL: 9,
    SERIES_LABEL: 8,
    OP_LABEL: 6,
    HEALTH_PILL_STAT_LABEL: 4,
  },
};

export const SVG_NAMESPACE = 'http://www.w3.org/2000/svg';

/**
 * Given a container d3 selection, select a child element of a given tag and
 * class. If multiple children matches the tag and class name, returns only
 * the first one.
 *
 * @param container
 * @param tagName tag name.
 * @param className (optional) Class name or list of class names.
 * @return selection of the element, or an empty selection
 */
export function selectChild(
  container,
  tagName: string,
  className?: string | string[],
): d3.Selection<any, any, any, any> {
  let children = container.node().childNodes;
  for (let i = 0; i < children.length; i++) {
    let child = children[i];
    if (child.tagName === tagName) {
      if (className instanceof Array) {
        let hasAllClasses = true;
        for (let j = 0; j < className.length; j++) {
          hasAllClasses = hasAllClasses && child.classList.contains(className[j]);
        }
        if (hasAllClasses) {
          return d3.select(child);
        }
      } else if (!className || child.classList.contains(className)) {
        return d3.select(child);
      }
    }
  }
  return d3.select(null);
}

/**
 * Given a container d3 selection, select a child svg element of a given tag
 * and class if exists or append / insert one otherwise.  If multiple children
 * matches the tag and class name, returns only the first one.
 *
 * @param container
 * @param tagName tag name.
 * @param className (optional) Class name or a list of class names.
 * @param before (optional) reference DOM node for insertion.
 * @return selection of the element
 */
export function selectOrCreateChild(
  container,
  tagName: string,
  className?: string | string[],
  before?,
): d3.Selection<any, any, any, any> {
  let child = selectChild(container, tagName, className);
  if (!child.empty()) {
    return child;
  }
  let newElement = document.createElementNS(SVG_NAMESPACE, tagName);
  if (className instanceof Array) {
    for (let i = 0; i < className.length; i++) {
      newElement.classList.add(className[i]);
    }
  } else {
    newElement.classList.add(className ?? '');
  }
  if (before) {
    // if before exists, insert
    container.node().insertBefore(newElement, before);
  } else {
    // otherwise, append
    container.node().appendChild(newElement);
  }
  return (
    d3
      .select(newElement)
      // need to bind data to emulate d3_selection.append
      .datum(container.datum())
  );
}

/** The minimum stroke width of an edge. */
export const MIN_EDGE_WIDTH = 0.75;
/** The maximum stroke width of an edge. */
export const MAX_EDGE_WIDTH = 12;
/** The exponent used in the power scale for edge thickness. */
const EDGE_WIDTH_SCALE_EXPONENT = 0.3;
/** The domain (min and max value) for the edge width. */
const DOMAIN_EDGE_WIDTH_SCALE = [1, 5000000];
export const EDGE_WIDTH_SIZE_BASED_SCALE: d3.ScalePower<number, number> = d3
  .scalePow()
  .exponent(EDGE_WIDTH_SCALE_EXPONENT)
  .domain(DOMAIN_EDGE_WIDTH_SCALE)
  .range([MIN_EDGE_WIDTH, MAX_EDGE_WIDTH])
  .clamp(true);

export const globalTooltips: { [key: string]: string } = {};
// NPU侧模型的节点前缀
export const NPU_PREFIX = 'N___';
// 标杆侧模型的节点前缀
export const BENCH_PREFIX = 'B___';
// 未匹配节点颜色
export const UNMATCHED_COLOR = '#C7C7C7';
// 展开对应侧节点
export const EXPAND_NODE = '展开对应侧节点';
// 数据发送
export const DATA_SEND = '数据发送';
// 数据接收
export const DATA_RECEIVE = '数据接收';
// 数据发送接收
export const DATA_SEND_RECEIVE = '数据发送接收';
// 数据读取时间
export const DATA_LOAD_TIME = 3000;
// 数据过大提示时间
export const DATA_NOTICE_TIME = 600;
