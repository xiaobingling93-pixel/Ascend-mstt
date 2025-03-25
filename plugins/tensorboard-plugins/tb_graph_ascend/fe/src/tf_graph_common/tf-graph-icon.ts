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

import { computed, customElement, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import { DarkModeMixin } from '../polymer/dark_mode_mixin';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import '../tf_dashboard_common/tensorboard-color';
import { MetanodeColors, OpNodeColors } from './render';

export enum GraphIconType {
  CONST = 'CONST',
  META = 'META',
  OP = 'OP',
  SERIES = 'SERIES',
  SUMMARY = 'SUMMARY',
  MULTI_COLLECTION = 'MULTI_COLLECTION',
  API_LIST = 'API_LIST',
}
@customElement('tf-graph-icon')
class TfGraphIcon extends LegacyElementMixin(DarkModeMixin(PolymerElement)) {
  static readonly template = html`
    <style>
      :host {
        font-size: 0;
      }

      :host(.dark-mode) svg {
        filter: invert(1);
      }

      .faded-rect {
        fill: url(#rectHatch);
      }

      .faded-ellipse {
        fill: url(#ellipseHatch);
      }

      .faded-rect,
      .faded-ellipse,
      .faded-series {
        stroke: var(--tb-graph-faded) !important;
      }
      #rectHatch line,
      #ellipseHatch line {
        color: #e0d4b3 !important;
        fill: white;
        stroke: #e0d4b3 !important;
      }
    </style>
    <!-- SVG for definitions -->
    <svg height="0" width="0" id="svgDefs">
      <defs>
        <ellipse id="op-node-stamp" rx="7.5" ry="3" stroke="inherit" fill="inherit"></ellipse>
      </defs>
    </svg>
    <template is="dom-if" if="[[_isType(type, 'OP')]]">
      <svg height$="[[height]]" preserveAspectRatio="xMinYMid meet" viewBox="0 0 16 8">
        <use
          xmlns:xlink="http://www.w3.org/1999/xlink"
          xlink:href="#op-node-stamp"
          fill$="[[_fill]]"
          stroke$="[[_stroke]]"
          class$="{{_fadedClass(faded, 'ellipse')}}"
          x="8"
          y="4"
        ></use>
      </svg>
    </template>
    <template is="dom-if" if="[[_isType(type, 'META')]]">
      <svg height$="[[height]]" preserveAspectRatio="xMinYMid meet" viewBox="0 0 37 16">
        <rect
          x="1"
          y="1"
          fill$="[[_fill]]"
          stroke$="[[_stroke]]"
          class$="{{_fadedClass(faded, 'rect')}}"
          stroke-width="2px"
          height="14"
          width="35"
          rx="5"
          ry="5"
        ></rect>
      </svg>
    </template>
    <template is="dom-if" if="[[_isType(type, 'MULTI_COLLECTION')]]">
      <svg height$="[[height]]" preserveAspectRatio="xMinYMid meet" viewBox="0 0 37 16">
        <rect
          x="1"
          y="1"
          fill$="[[_fill]]"
          stroke$="[[_stroke]]"
          class$="{{_fadedClass(faded, 'rect')}}"
          stroke-width="2px"
          stroke-dasharray="1, 1"
          height="14"
          width="35"
          rx="5"
          ry="5"
        ></rect>
      </svg>
    </template>
    <template is="dom-if" if="[[_isType(type, 'API_LIST')]]">
      <svg height$="[[height]]" preserveAspectRatio="xMinYMid meet" viewBox="0 0 37 16">
        <rect
          x="1"
          y="1"
          fill$="[[_fill]]"
          stroke$="[[_stroke]]"
          class$="{{_fadedClass(faded, 'rect')}}"
          stroke-width="2px"
          stroke-dasharray="7, 2"
          height="14"
          width="35"
          rx="5"
          ry="5"
        ></rect>
      </svg>
    </template>
  `;

  @property({ type: String })
  type: string;

  @property({ type: Boolean })
  vertical: boolean = false;

  @property({ type: String })
  fillOverride: string | null = null;

  @property({ type: String })
  strokeOverride: string | null = null;

  @property({ type: Number })
  height: number = 20;

  @property({ type: Boolean })
  faded: boolean = false;

  getSvgDefinableElement(): HTMLElement {
    return this.$.svgDefs as HTMLElement;
  }

  @computed('type', 'fillOverride')
  get _fill(): string {
    let type = this.type;
    let fillOverride = this.fillOverride;
    if (fillOverride != null) {
      return fillOverride;
    }
    switch (type) {
      case GraphIconType.META:
        return MetanodeColors.DEFAULT_FILL;
      default:
        return OpNodeColors.DEFAULT_FILL;
    }
  }

  @computed('type', 'strokeOverride')
  get _stroke(): string {
    let type = this.type;
    let strokeOverride = this.strokeOverride;
    if (strokeOverride != null) {
      return strokeOverride;
    }
    switch (type) {
      case GraphIconType.META:
        return MetanodeColors.DEFAULT_STROKE;
      default:
        return OpNodeColors.DEFAULT_STROKE;
    }
  }

  /**
   * Test whether the specified node's type, or the literal type string,
   * match a particular other type.
   */
  _isType(type: GraphIconType, targetType: GraphIconType): boolean {
    return type === targetType;
  }

  _fadedClass(faded: boolean, shape: string): string {
    return `${faded ? `faded-${shape}` : ''}`;
  }
}
