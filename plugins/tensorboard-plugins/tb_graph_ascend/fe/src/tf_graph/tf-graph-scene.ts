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

Copyright (c) 2025, Huawei Technologies.
Adapt to the model hierarchical visualization data collected by the msprobe tool
==============================================================================*/

import { customElement, observe, property } from '@polymer/decorators';
import { PolymerElement } from '@polymer/polymer';
import * as d3 from 'd3';
import * as _ from 'lodash';
import { DarkModeMixin } from '../polymer/dark_mode_mixin';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import * as tb_debug from '../tb_debug';
import '../tf_dashboard_common/tensorboard-color';
import * as tf_graph from '../tf_graph_common/graph';
import * as tf_graph_layout from '../tf_graph_common/layout';
import * as tf_graph_minimap from '../tf_graph_common/minimap';
import * as tf_graph_scene_node from '../tf_graph_common/node';
import * as tf_graph_render from '../tf_graph_common/render';
import * as tf_graph_scene from '../tf_graph_common/scene';
import { TfGraphScene } from '../tf_graph_common/tf-graph-scene';
import * as tf_graph_util from '../tf_graph_common/util';
import './tf-graph-minimap';
import { template } from './tf-graph-scene.html';

// 限制滚动速度
const MOUSE_MOVE_DELTA = 800;
@customElement('tf-graph-scene')
class TfGraphScene2 extends LegacyElementMixin(DarkModeMixin(PolymerElement)) implements TfGraphScene {
  static readonly template = template;
  @property({ type: Number })
  _step: number = 20;

  @property({ type: Number })
  mouseX: number = 0;

  @property({ type: Number })
  mouseY: number = 0;

  @property({ type: Number })
  x: number = 0;

  @property({ type: Number })
  y: number = 0;

  @property({ type: Object })
  renderHierarchy: tf_graph_render.RenderGraphInfo;

  @property({ type: String })
  name: string;

  // For each render hierarchy, we only fit it to the viewport once (when the scene is attached to
  // the DOM). We do not fit the hierarchy again (unless the user clicks the reset button). For
  // instance, if the user enters a certain view in the graph, switches to another dashboard, and
  // returns to the graph dashboard, the user expects the previous view. These properties enable
  // that behavior.

  /** Whether the scene has fit the current render hierarchy (to the viewport) at least once. */
  @property({ type: Boolean })
  _hasRenderHierarchyBeenFitOnce: boolean;

  /** Whether this scene element is currently attached to a parent element. */
  @property({ type: Boolean })
  _isAttached: boolean;

  /** This property is a d3_zoom object. */
  @property({ type: Object })
  _zoom: object;

  /** This property is a d3_drag object. */
  @property({ type: Object })
  _drag: object;

  @property({
    type: String,
    observer: '_highlightedNodeChanged',
  })
  highlightedNode: string;

  @property({
    type: String,
    observer: '_selectedNodeChanged',
  })
  selectedNode: string;

  @property({
    type: String,
    observer: '_linkedNodeChanged',
  })
  linkedNode: string;

  /** Keeps track of if the graph has been zoomed/panned since loading */
  @property({
    type: Boolean,
    observer: '_onZoomChanged',
  })
  _zoomed: boolean = false;

  /**
   * Keeps track of the starting coordinates of a graph zoom/pan.
   *
   * @private {{x: number, y: number}?}
   */
  @property({
    type: Object,
  })
  _zoomStartCoords: { x: number; y: number } | null = null;

  /**
   * Keeps track of the current coordinates of a graph zoom/pan
   *
   * @private {{x: number, y: number}?}
   */
  @property({
    type: Object,
  })
  _zoomTransform: { x: number; y: number } | null = null;

  /** Maximum distance of a zoom event for it to be interpreted as a click */
  @property({
    type: Number,
  })
  _maxZoomDistanceForClick: number = 20;

  /*
   * Dictionary for easily stylizing nodes when state changes.
   * _nodeGroupIndex[nodeName] = d3_selection of the nodeGroup
   */
  @property({
    type: Object,
  })
  _nodeGroupIndex = {};

  /*
   * Dictionary for easily stylizing annotation nodes when state changes.
   * _annotationGroupIndex[nodeName][hostNodeName] =
   *   d3_selection of the annotationGroup
   */
  @property({
    type: Object,
  })
  _annotationGroupIndex = {};

  /*
   * Dictionary for easily stylizing edges when state changes.
   * _edgeGroupIndex[edgeName] = d3_selection of the edgeGroup
   */
  @property({
    type: Object,
  })
  _edgeGroupIndex = {};

  /**
   * Max font size for metanode label strings.
   */
  @property({
    type: Number,
  })
  maxMetanodeLabelLengthFontSize: number = 9;

  /**
   * Min font size for metanode label strings.
   */
  @property({ type: Number })
  minMetanodeLabelLengthFontSize: number = 6;

  /**
   * Metanode label strings longer than this are given smaller fonts.
   */
  @property({ type: Number })
  maxMetanodeLabelLengthLargeFont: number = 11;

  /**
   * Metanode label strings longer than this are truncated with ellipses.
   */
  @property({ type: Number })
  maxMetanodeLabelLength: number = 50;

  @property({ type: Object })
  progress: any;

  // An array of ContextMenuItem objects. Items that appear in the context
  // menu for a node.
  @property({ type: Array })
  nodeContextMenuItems: unknown[];

  @property({ type: Boolean })
  showMinimap: boolean = true;

  /**
   * A minimap object to notify for zoom events.
   */
  private minimap: tf_graph_minimap.Minimap;

  private enablePanSignal: Boolean = true;

  @observe('renderHierarchy')
  _renderHierarchyChanged(): void {
    let renderHierarchy = this.renderHierarchy;
    this._hasRenderHierarchyBeenFitOnce = false;
    this._resetState();
    this._build(renderHierarchy);
  }

  @observe('showMinimap')
  _minimapVisChanged(): void {
    const minimap = this.$.minimap as HTMLElement;
    minimap.style.display = this.showMinimap ? 'block' : 'none';
  }

  // Animation and fitting must come after the observer for the hierarchy changing because we must
  // first build the render hierarchy.
  @observe('_isAttached', 'renderHierarchy')
  _animateAndFit(): void {
    const isAttached = this._isAttached;
    if (this._hasRenderHierarchyBeenFitOnce || !isAttached) {
      // Do not animate and fit if the scene has already fitted this render hierarchy once. Or if
      // the graph dashboard is not attached (in which case the scene lacks DOM info for fitting).
      return;
    }
    // Fit to screen after the graph is done animating.
    setTimeout(this.fit.bind(this), tf_graph_layout.PARAMS.animation.duration);
  }

  getNode(nodeName): tf_graph_render.RenderNodeInfo {
    return this.renderHierarchy.getRenderNodeByName(nodeName);
  }

  isNodeExpanded(node): boolean {
    return node.expanded;
  }

  setNodeExpanded(renderNode): void {
    this._build(this.renderHierarchy);
    this._updateLabels(!this._zoomed);
  }

  /**
   * Pans to a node. Assumes that the node exists.
   * @param nodeName {string} The name of the node to pan to.
   */
  panToNode(nodeName): void {
    const zoomed = tf_graph_scene.panToNode(nodeName, this.$.svg, this.$.root, this._zoom);
    if (zoomed) {
      this._zoomed = true;
    }
  }

  /**
   * Returns the outer-most SVG that renders the graph.
   */
  getGraphSvgRoot(): SVGElement {
    return this.$.svg as SVGElement;
  }

  getContextMenu(): HTMLElement {
    return this.$.contextMenu as HTMLElement;
  }

  /**
   * Resets the state of the component. Called whenever the whole graph
   * (dataset) changes.
   */
  _resetState(): void {
    // Reset the state of the component.
    this._nodeGroupIndex = {};
    this._annotationGroupIndex = {};
    this._edgeGroupIndex = {};
    this._updateLabels(false);
    // Remove all svg elements under the 'root' svg group.
    d3.select(this.$.svg).select('#root').selectAll('*').remove();
    // And the defs.
    tf_graph_scene_node.removeGradientDefinitions(this.$.svg as SVGElement);
  }

  /** Main method for building the scene */
  _build(renderHierarchy: tf_graph_render.RenderGraphInfo): void {
    if (!renderHierarchy) {
      return;
    }
    tf_graph_util.time(
      'tf-graph-scene (layout):',
      (): void => {
        // layout the scene for this meta / series node
        tf_graph_layout.layoutScene(renderHierarchy.root);
      },
      tb_debug.GraphDebugEventId.RENDER_SCENE_LAYOUT,
    );
    tf_graph_util.time(
      'tf-graph-scene (build scene):',
      (): void => {
        tf_graph_scene_node.buildGroupForScene(d3.select(this.$.root), renderHierarchy.root, this);
        tf_graph_scene.addGraphClickListener(this.$.svg, this);
      },
      tb_debug.GraphDebugEventId.RENDER_SCENE_BUILD_SCENE,
    );
    // Update the minimap again when the graph is done animating.
    setTimeout((): void => {
      this.minimap.update();
    }, tf_graph_layout.PARAMS.animation.duration);
  }

  ready(): void {
    super.ready();

    this.addEventListener('no-pan-to-node', this._noPanToNode.bind(this))
    this._zoom = d3
      .zoom()
      .on('end', () => {
        if (this._zoomStartCoords && this._zoomTransform) {
          // Calculate the total distance dragged during the zoom event.
          // If it is sufficiently small, then fire an event indicating
          // that zooming has ended. Otherwise wait to fire the zoom end
          // event, so that a mouse click registered as part of this zooming
          // is ignored (as this mouse click was part of a zooming, and should
          // not be used to indicate an actual click on the graph).
          let dragDistance = Math.sqrt(
            Math.pow(this._zoomStartCoords.x - this._zoomTransform.x, 2) +
            Math.pow(this._zoomStartCoords.y - this._zoomTransform.y, 2),
          );
          if (dragDistance < this._maxZoomDistanceForClick) {
            this._fireEnableClick();
          } else {
            setTimeout(this._fireEnableClick.bind(this), 50);
          }
        }
        this._zoomStartCoords = null;
      }
      )
      .on('zoom', () => {
        this._zoomTransform = d3.event.transform;
        if (!this._zoomStartCoords) {
          this._zoomStartCoords = this._zoomTransform;
          this.fire('disable-click');
        }
        this._zoomed = true;
        d3.select(this.$.root).attr('transform', d3.event.transform.toString());
        this.x = d3.event.transform.x;
        this.y = d3.event.transform.y;
        // Notify the minimap.
        this.minimap.zoom(d3.event.transform);
      });

    d3.select(this.$.svg).call(this._addEventListener.bind(this)).on('dblclick.zoom', null);
    d3.select(window).on('resize', () => {
      // Notify the minimap that the user's window was resized.
      // The minimap will figure out the new dimensions of the main svg
      // and will use the existing translate and scale params.
      this.minimap.zoom();
    });
    // Initialize the minimap.
    this.minimap = (this.$.minimap as any).init(
      this.$.svg,
      this.$.root,
      this._zoom,
      tf_graph_layout.PARAMS.minimap.size,
      tf_graph_layout.PARAMS.subscene.meta.labelHeight,
    );

    // Add keyboard event listener
    this._addEventListener();
  }

  _addEventListener(): void {
    let isDragging = false;
    let startX;
    let startY;
    let lastTime = 0;
    const smoothFactor = 0.2; // 控制平滑的因子
    const maxDelta = MOUSE_MOVE_DELTA; // 限制滚动速度
    const svgElement = this.$.svg as SVGSVGElement;
    svgElement.setAttribute('tabindex', '0');

    svgElement.addEventListener('mousedown', (event: MouseEvent) => {
      isDragging = true;
      startX = event.clientX;
      startY = event.clientY;
      svgElement.focus();
    });
    window.addEventListener('mouseup', () => {
      isDragging = false;
    });
    svgElement.addEventListener('mousemove', (event: MouseEvent) => {
      [this.mouseX, this.mouseY] = [event.clientX, event.clientY];
    });
    // prettier-ignore
    svgElement.addEventListener(
      'mousemove', // 不能根据鼠标移动来ws，提节流方法
      _.throttle((event: MouseEvent) => {
        if (isDragging) {
          this.x = this.x + ((event.clientX - startX) / 2);
          this.y = this.y + ((event.clientY - startY) / 2);
          this._moveView();
          startX = event.clientX;
          startY = event.clientY;
        }
      }, 15),
    ); // 节流，限制触发频率。每15毫秒，事件最多执行一次
    // prettier-ignore
    svgElement.addEventListener('wheel', (event: WheelEvent) => {
      const currentTime = performance.now();
      const deltaTime = currentTime - lastTime;
      if (deltaTime > 16) {
        // 确保每帧调用
        const deltaY = Math.sign(event.deltaY) * Math.min(Math.abs(event.deltaY), maxDelta);
        this.y = this.y - (deltaY * smoothFactor * 2);
        this._moveView();
        lastTime = currentTime;
      }
    });
    svgElement.addEventListener('keydown', (event: KeyboardEvent) => {
      switch (event.key) {
        case 'w':
        case 'W':
          this._scaleView(1.1);
          break;
        case 's':
        case 'S':
          this._scaleView(0.9);
          break;
        case 'a':
        case 'A':
          this.x += this._step;
          this._moveView();
          break;
        case 'd':
        case 'D':
          this.x -= this._step;
          this._moveView();
          break;
        default:
          return; // Exit if it's not an arrow key
      }
    });
  }

  _scaleView(scaleFactor: number): void {
    if (this._zoomTransform) {
      const svgElement = this.$.svg as SVGSVGElement;
      const currentTransform = d3.zoomTransform(svgElement);
      const k = currentTransform.k === 0 ? 1 : currentTransform.k;
      const [mouseX, mouseY] = [
        this.mouseX - svgElement.getBoundingClientRect().left,
        this.mouseY - svgElement.getBoundingClientRect().top,
      ];
      const translateX = (mouseX - currentTransform.x) / k;
      const translateY = (mouseY - currentTransform.y) / k;
      const newScale = currentTransform.k * scaleFactor;
      this.x = mouseX - (translateX * newScale);
      this.y = mouseY - (translateY * newScale);
      const newTransform = d3.zoomIdentity.translate(this.x, this.y).scale(newScale);
      d3.select(this.$.svg).call(d3.zoom().transform, newTransform);
      d3.select(this.$.root).attr('transform', newTransform.toString());
      this._zoomTransform = newTransform;
      this.minimap.zoom(newTransform);
    }
  }

  _moveView(): void {
    if (this._zoomTransform) {
      requestAnimationFrame(() => {
        const svgElement = this.$.svg as SVGElement;
        const currentTransform = d3.zoomTransform(svgElement);
        const newTransform = d3.zoomIdentity.translate(this.x, this.y).scale(currentTransform.k);
        const svgSelection = d3.select(this.$.svg);
        const rootSelection = d3.select(this.$.root);
        svgSelection.call(d3.zoom().transform, newTransform);
        rootSelection.attr('transform', newTransform.toString());
        // 更新存储的变换对象
        this._zoomTransform = newTransform;
        // 通知小地图更新，只在变换发生变化时调用
        this.minimap.zoom(newTransform);
      });
    }
  }

  override attached(): void {
    this.set('_isAttached', true);
  }

  override detached(): void {
    this.set('_isAttached', false);
  }

  _updateLabels(showLabels): void {
    let mainGraphTitleElement = this.$$('.title') as HTMLElement;
    let titleStyle = mainGraphTitleElement.style;
    let auxTitleElement = this.$$('.auxTitle') as HTMLElement;
    let auxTitleStyle = auxTitleElement.style;
    let functionLibraryTitleStyle = (this.$$('.functionLibraryTitle') as HTMLElement).style;
    const root = d3.select(this.$.svg);
    let core = root.select(`.${tf_graph_scene.Class.Scene.GROUP}>.${tf_graph_scene.Class.Scene.CORE}`).node();
    const isProgressComplete = showLabels && this.progress && this.progress.value === 100 && core;
    if (isProgressComplete) {
      let aux =
        root.select(`.${tf_graph_scene.Class.Scene.GROUP}>.${tf_graph_scene.Class.Scene.INEXTRACT}`).node() ||
        root.select(`.${tf_graph_scene.Class.Scene.GROUP}>.${tf_graph_scene.Class.Scene.OUTEXTRACT}`).node();
      let coreX = (core as any).getCTM().e;
      let auxX = aux ? (aux as any).getCTM().e : null;
      titleStyle.display = 'inline';
      titleStyle.left = `${coreX}px`;
      if (auxX !== null && auxX !== coreX) {
        auxTitleStyle.display = 'inline';
        // Make sure that the aux title is positioned rightwards enough so as to
        // prevent overlap with the main graph title.
        auxX = Math.max(coreX + mainGraphTitleElement.getBoundingClientRect().width, auxX);
        auxTitleStyle.left = `${auxX}px`;
      } else {
        auxTitleStyle.display = 'none';
      }
      let functionLibrary = root
        .select(`.${tf_graph_scene.Class.Scene.GROUP}>.${tf_graph_scene.Class.Scene.FUNCTION_LIBRARY}`)
        .node();
      let functionLibraryX = functionLibrary ? (functionLibrary as any).getCTM().e : null;
      if (functionLibraryX !== null && functionLibraryX !== auxX) {
        functionLibraryTitleStyle.display = 'inline';
        // Make sure that the function library title is positioned rightwards
        // enough so as to prevent overlap with other content.
        functionLibraryX = Math.max(auxX + auxTitleElement.getBoundingClientRect().width, functionLibraryX);
        functionLibraryTitleStyle.left = `${functionLibraryX}px`;
      } else {
        functionLibraryTitleStyle.display = 'none';
      }
    } else {
      titleStyle.display = 'none';
      auxTitleStyle.display = 'none';
      functionLibraryTitleStyle.display = 'none';
    }
  }

  fit(): void {
    this._hasRenderHierarchyBeenFitOnce = true;
    tf_graph_scene.fit(
      this.$.svg,
      this.$.root,
      this._zoom,
      (): void => {
        this._zoomed = false;
      },
    );
  }

  isNodeSelected(n): boolean {
    return n === this.selectedNode;
  }

  isNodeHighlighted(n): boolean {
    return n === this.highlightedNode;
  }

  isNodeLinked(n): boolean {
    return n === this.linkedNode;
  }

  addAnnotationGroup(a, d, selection): void {
    let an = a.node.name;
    this._annotationGroupIndex[an] = this._annotationGroupIndex[an] || {};
    this._annotationGroupIndex[an][d.node.name] = selection;
  }

  getAnnotationGroupsIndex(a): any {
    return this._annotationGroupIndex[a];
  }

  removeAnnotationGroup(a, d): void {
    delete this._annotationGroupIndex[a.node.name][d.node.name];
  }

  addNodeGroup(n, selection): void {
    this._nodeGroupIndex[n] = selection;
  }

  getNodeGroup(n): any {
    return this._nodeGroupIndex[n];
  }

  removeNodeGroup(n): void {
    delete this._nodeGroupIndex[n];
  }

  /**
   * Update node and annotation node of the given name.
   * @param  {String} n node name
   */
  _updateNodeState(n): void {
    let node = this.getNode(n);
    if (!node) {
      return;
    }
    let nodeGroup = this.getNodeGroup(n);
    if (nodeGroup) {
      tf_graph_scene_node.stylize(nodeGroup, node, this as any);
    }
    let annotationGroupIndex = this.getAnnotationGroupsIndex(n);
    _.each(annotationGroupIndex, (aGroup, hostName) => {
      tf_graph_scene_node.stylize(aGroup, node, this as any, tf_graph_scene.Class.Annotation.NODE);
    });
  }

  /**
   * Handles new node selection. 1) Updates the selected-state of each node,
   * 2) triggers input tracing.
   * @param selectedNode {string} The name of the newly selected node.
   * @param oldSelectedNode {string} The name of the previously selected node.
   * @private
   */
  _selectedNodeChanged(selectedNode, oldSelectedNode): void {
    if (selectedNode === oldSelectedNode) {
      return;
    }
    if (oldSelectedNode) {
      this._updateNodeState(oldSelectedNode);
    }
    if (!selectedNode) {
      this.set('linkedNode', '');
      return;
    }
    let node = this.renderHierarchy.hierarchy.node(selectedNode);
    if (!node) {
      return;
    }
    // Update the minimap to reflect the highlighted (selected) node.
    (this.minimap as any).update();
    let nodeParents: string[] = [];
    // Create list of all metanode parents of the selected node.
    while (node.parentNode !== null && node.parentNode.name !== tf_graph.ROOT_NAME) {
      node = (node as any).parentNode;
      nodeParents.push(node.name);
    }
    // Ensure each parent metanode is built and expanded.
    let topParentNodeToBeExpanded;
    _.forEachRight(nodeParents, (parentName) => {
      this.renderHierarchy.buildSubhierarchy(parentName);
      let renderNode = this.renderHierarchy.getRenderNodeByName(parentName);
      if (renderNode.node.isGroupNode && !renderNode.expanded) {
        renderNode.expanded = true;
        if (!topParentNodeToBeExpanded) {
          topParentNodeToBeExpanded = renderNode;
        }
      }
    });
    // If any expansion was needed to display this selected node, then
    // inform the scene of the top-most expansion.
    if (topParentNodeToBeExpanded) {
      this.setNodeExpanded(topParentNodeToBeExpanded);
      this._zoomed = true;
    }
    if (selectedNode) {
      this._updateNodeState(selectedNode);
    }
    // Give time for any expanding to finish before panning to a node.
    // Otherwise, the pan will be computed from incorrect measurements.
    setTimeout(() => {
      // 鼠标点击不自动移动居中
      if (this.enablePanSignal) {
        this.panToNode(selectedNode);
      }
      this.enablePanSignal = true;
    }, tf_graph_layout.PARAMS.animation.duration);
  }

  _highlightedNodeChanged(highlightedNode, oldHighlightedNode): void {
    if (highlightedNode === oldHighlightedNode) {
      return;
    }
    if (highlightedNode) {
      this._updateNodeState(highlightedNode);
    }
    if (oldHighlightedNode) {
      this._updateNodeState(oldHighlightedNode);
    }
  }

  _linkedNodeChanged(linkedNode, oldLinkedNode): void {
    if (linkedNode === oldLinkedNode) {
      return;
    }
    if (oldLinkedNode) {
      this._updateNodeState(oldLinkedNode);
    }
    if (linkedNode) {
      this._updateNodeState(linkedNode);
    }
  }

  _onZoomChanged(): void {
    this._updateLabels(!this._zoomed);
  }

  _fireEnableClick(): void {
    this.fire('enable-click');
  }
  
  // 取消鼠标点击自动居中
  _noPanToNode(): void {
    this.enablePanSignal = false
  }
}
