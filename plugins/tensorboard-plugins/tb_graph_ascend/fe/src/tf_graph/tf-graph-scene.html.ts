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
==============================================================================*/
import { html } from '@polymer/polymer';

// Please keep node font-size/classnames in sync with tf-graph-common/common.ts
export const template = html`
  <style>
    :host(.dark-mode) {
      filter: invert(1);
    }

    :host {
      display: flex;
      font-size: 20px;
      height: 100%;
      width: 100%;
    }

    #svg {
      flex: 1;
      font-family: Roboto, sans-serif;
      height: 100%;
      overflow: hidden;
      width: 100%;
      outline: none;
    }

    #hidden {
      position: fixed;
      top: 0px;
      visibility: hidden;
    }

    text {
      user-select: none;
    }

    /* --- Node and annotation-node for Metanode --- */

    .meta > .nodeshape > rect,
    .meta > .annotation-node > rect {
      cursor: pointer;
      fill: hsl(0, 0%, 70%);
    }
    .node.meta.highlighted > .nodeshape > rect,
    .node.meta.highlighted > .annotation-node > rect {
      stroke-width: 3;
    }
    .annotation.meta.highlighted > .nodeshape > rect,
    .annotation.meta.highlighted > .annotation-node > rect {
      stroke-width: 3;
    }
    .meta.selected > .nodeshape > rect,
    .meta.selected > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 3;
    }
    .node.meta.selected.expanded > .nodeshape > rect,
    .node.meta.selected.expanded > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 3;
    }
    .annotation.meta.selected > .nodeshape > rect,
    .annotation.meta.selected > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 3;
    }
    .node.meta.selected.expanded.highlighted > .nodeshape > rect,
    .node.meta.selected.expanded.highlighted > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 4;
    }
    .meta.linked > .nodeshape > rect,
    .meta.linked > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 3;
    }
    .node.meta.linked.expanded > .nodeshape > rect,
    .node.meta.linked.expanded > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 3;
    }
    .annotation.meta.linked > .nodeshape > rect,
    .annotation.meta.linked > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 3;
    }
    .node.meta.linked.expanded.highlighted > .nodeshape > rect,
    .node.meta.linked.expanded.highlighted > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 4;
    }
    .api_list > .nodeshape > rect,
    .api_list > .annotation-node > rect {
      cursor: pointer;
      fill: hsl(0, 0%, 70%);
      stroke-width: 1;
      stroke-dasharray: 15, 2;
    }
    .node.api_list.highlighted > .nodeshape > rect,
    .node.api_list.highlighted > .annotation-node > rect {
      stroke-width: 2;
    }
    .annotation.api_list.highlighted > .nodeshape > rect,
    .annotation.api_list.highlighted > .annotation-node > rect {
      stroke-width: 1;
    }
    .api_list.selected > .nodeshape > rect,
    .api_list.selected > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 2;
    }
    .node.api_list.selected.expanded > .nodeshape > rect,
    .node.api_list.selected.expanded > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 3;
    }
    .annotation.api_list.selected > .nodeshape > rect,
    .annotation.api_list.selected > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 2;
    }
    .node.api_list.selected.expanded.highlighted > .nodeshape > rect,
    .node.api_list.selected.expanded.highlighted > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 4;
    }
    .api_list.linked > .nodeshape > rect,
    .api_list.linked > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 2;
    }
    .node.api_list.linked.expanded > .nodeshape > rect,
    .node.api_list.linked.expanded > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 3;
    }
    .annotation.api_list.linked > .nodeshape > rect,
    .annotation.api_list.linked > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 2;
    }
    .node.api_list.linked.expanded.highlighted > .nodeshape > rect,
    .node.api_list.linked.expanded.highlighted > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 4;
    }

    .multi_collection > .nodeshape > rect,
    .multi_collection > .annotation-node > rect {
      cursor: pointer;
      fill: hsl(0, 0%, 70%);
      stroke-width: 1;
      stroke-dasharray: 1, 1;
    }
    .node.multi_collection.highlighted > .nodeshape > rect,
    .node.multi_collection.highlighted > .annotation-node > rect {
      stroke-width: 2;
    }
    .annotation.multi_collection.highlighted > .nodeshape > rect,
    .annotation.multi_collection.highlighted > .annotation-node > rect {
      stroke-width: 1;
    }
    .multi_collection.selected > .nodeshape > rect,
    .multi_collection.selected > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 2;
    }
    .node.multi_collection.selected.expanded > .nodeshape > rect,
    .node.multi_collection.selected.expanded > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 3;
    }
    .annotation.multi_collection.selected > .nodeshape > rect,
    .annotation.multi_collection.selected > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 2;
    }
    .node.multi_collection.selected.expanded.highlighted > .nodeshape > rect,
    .node.multi_collection.selected.expanded.highlighted > .annotation-node > rect {
      stroke: #4058d1;
      stroke-width: 4;
    }
    .multi_collection.linked > .nodeshape > rect,
    .multi_collection.linked > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 2;
    }
    .node.multi_collection.linked.expanded > .nodeshape > rect,
    .node.multi_collection.linked.expanded > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 3;
    }
    .annotation.multi_collection.linked > .nodeshape > rect,
    .annotation.multi_collection.linked > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 2;
    }
    .node.multi_collection.linked.expanded.highlighted > .nodeshape > rect,
    .node.multi_collection.linked.expanded.highlighted > .annotation-node > rect {
      stroke: #7d8dde !important;
      stroke-width: 4;
    }

    /* --- Op Node --- */

    .op > .nodeshape > .nodecolortarget,
    .op > .annotation-node > .nodecolortarget {
      cursor: pointer;
      fill: #fff;
      stroke: #ccc;
    }

    .op.selected > .nodeshape > .nodecolortarget,
    .op.selected > .annotation-node > .nodecolortarget {
      stroke: #4058d1;
      stroke-width: 2;
    }

    .op.highlighted > .nodeshape > .nodecolortarget,
    .op.highlighted > .annotation-node > .nodecolortarget {
      stroke-width: 2;
    }

    .op.linked > .nodeshape > .nodecolortarget,
    .op.linked > .annotation-node > .nodecolortarget {
      stroke: #7d8dde !important;
      stroke-width: 2;
    }

    /* --- Node label --- */

    .node {
      transform: translateZ(1px);
    }

    .node > text.nodelabel {
      cursor: pointer;
      fill: #444;
      font-size: 9px;
    }

    .meta.expanded > text.nodelabel {
      font-size: 9px;
    }

    .op > text.nodelabel {
      font-size: 6px;
    }

    .node.meta.expanded > text.nodelabel {
      cursor: normal;
    }

    .node.multi_collection.expanded > text.nodelabel {
      cursor: normal;
    }

    .node.api_list.expanded > text.nodelabel {
      cursor: normal;
    }

    .annotation.meta.highlighted > text.annotation-label {
      fill: #50a3f7;
    }

    .annotation.meta.selected > text.annotation-label {
      fill: #4285f4;
    }

    /* --- Annotation --- */

    /* only applied for annotations that are not summary or constant.
    (.summary, .constant gets overridden below) */
    .annotation > .annotation-node > * {
      stroke-width: 0.5;
      stroke-dasharray: 1, 1;
    }

    /* Hide annotations on expanded meta nodes since they're redundant. */
    .expanded > .in-annotations,
    .expanded > .out-annotations {
      display: none;
    }

    /* --- Group node expand/collapse button --- */

    /* Hides expand/collapse buttons when a node isn't expanded or highlighted. Using
       incredibly small opacity so that the bounding box of the <g> parent still takes
       this container into account even when it isn't visible */
    .node:not(.highlighted):not(.expanded) > .nodeshape > .buttoncontainer {
      opacity: 0.01;
    }
    .node.highlighted > .nodeshape > .buttoncontainer {
      cursor: pointer;
    }
    .expandbutton,
    .collapsebutton {
      stroke: white;
    }
    /* Do not let the path elements in the button take pointer focus */
    .node > .nodeshape > .buttoncontainer > .expandbutton,
    .node > .nodeshape > .buttoncontainer > .collapsebutton {
      pointer-events: none;
    }
    /* Only show the expand button when a node is collapsed and only show the
       collapse button when a node is expanded. */
    .node.expanded > .nodeshape > .buttoncontainer > .expandbutton {
      display: none;
    }
    .node:not(.expanded) > .nodeshape > .buttoncontainer > .collapsebutton {
      display: none;
    }

    .titleContainer {
      position: relative;
      top: 20px;
    }

    .title,
    .auxTitle,
    .functionLibraryTitle {
      position: absolute;
    }

    #minimap {
      position: absolute;
      right: 20px;
      top: 20px;
    }

    .context-menu {
      position: absolute;
      display: none;
      background-color: #e2e2e2;
      border-radius: 2px;
      font-size: 14px;
      min-width: 150px;
      border: 1px solid #d4d4d4;
    }

    .context-menu ul {
      list-style-type: none;
      margin: 0;
      padding: 0;
      cursor: default;
    }

    .context-menu ul li {
      padding: 4px 16px;
    }

    .context-menu ul li:hover {
      background-color: #f3913e;
      color: white;
    }
  </style>
  <div class="titleContainer">
    <div id="title" class="title"></div>
    <div id="auxTitle" class="auxTitle"></div>
    <div id="functionLibraryTitle" class="functionLibraryTitle"></div>
  </div>
  <svg id="svg">
    <rect fill="white" width="10000" height="10000"></rect>
    <g id="root"></g>
  </svg>
  <tf-graph-minimap id="minimap"></tf-graph-minimap>
  <div id="contextMenu" class="context-menu"></div>
`;