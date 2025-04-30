/* Copyright (c) 2025, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { PolymerElement, html } from '@polymer/polymer';
import { customElement } from '@polymer/decorators';
@customElement('scene-legend')
class Legend extends PolymerElement {
  static get template(): HTMLTemplateElement {
    return html`
      <style>
        .legend {
          display: flex;
          justify-content: center;
          align-items: center;
          margin-top: 10px;
          background: #fff;
          height: 40px;
        }
        .legend-item {
          margin-right: 10px;
          display: flex;
          align-items: center;
        }
        .legend-item-value {
          margin-left: 5px;
          font-size: 12px;
        }
        .legend-clarifier {
          color: #266236;
          cursor: pointer;
          height: 16px;
          width: 16px;
          margin-left: 4px;
          display: inline-block;
          text-decoration: underline;
          background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDIwMDAgMjAwMCI+PHBhdGggZmlsbD0iIzc5Nzk3OSIgZD0iTTk3MSAxMzY4cTE4IDAgMzEgMTIgMTIgMTMgMTIgMzF2MTQzcTAgMTgtMTIgMzEtMTMgMTItMzEgMTJIODI3cS0xNyAwLTMwLTEyLTEzLTEzLTEzLTMxdi0xNDNxMC0xOCAxMy0zMSAxMy0xMiAzMC0xMmgxNDR6bTEyMi03NDJxODYgNDMgMTM4IDExNSA1MiA3MSA1MiAxNjEgMCA5Ny01NyAxNjUtMzYgNDAtMTE5IDg3LTQzIDI1LTU3IDM5LTI1IDE4LTI1IDQzdjU5SDc3NHYtNzBxMC04MiA1Ny0xNDAgMzItMzIgMTA4LTc1bDctNHE1NC0zMiA3NS01MCAyNS0yNSAyNS01NyAwLTQzLTQ0LTcyLTQ1LTI5LTEwMS0yOXQtOTUgMjVxLTMyIDIyLTc5IDgzLTExIDExLTI3IDE0LTE2IDQtMzAtN2wtMTAxLTc1cS0xNC0xMS0xNi0yOXQ1LTMycTY4LTk3IDE1NS0xNDYgODYtNDggMjA4LTQ4IDg2IDAgMTcyIDQzem01NzYgMjlxMTIxIDIwNCAxMjEgNDQ1IDAgMjQwLTEyMSA0NDUtMTIwIDIwNC0zMjUgMzI1LTIwNCAxMjAtNDQ1IDEyMC0yNDAgMC00NDUtMTIwLTIwNC0xMjEtMzI1LTMyNVE5IDEzNDAgOSAxMTAwcTAtMjQxIDEyMC00NDUgMTIxLTIwNSAzMjUtMzI1IDIwNS0xMjEgNDQ1LTEyMSAyNDEgMCA0NDUgMTIxIDIwNSAxMjAgMzI1IDMyNXptLTE0OSA4MDRxOTctMTY1IDk3LTM1OXQtOTctMzU5cS05Ny0xNjUtMjYyLTI2MnQtMzU5LTk3cS0xOTQgMC0zNTkgOTdUMjc4IDc0MXEtOTcgMTY1LTk3IDM1OXQ5NyAzNTlxOTcgMTY1IDI2MiAyNjJ0MzU5IDk3cTE5NCAwIDM1OS05N3QyNjItMjYyeiIvPjwvc3ZnPg==');
        }

        .legend-clarifier paper-tooltip {
          width: 150px;
        }

        .custom-tooltip {
          font-size: 14px;
        }
      </style>
      <div class="legend">
        <div class="legend-item">
          <tf-graph-icon type="META" height="16" fill-override="#eee" stroke-override="#a6a6a6"></tf-graph-icon>
          <span class="legend-item-value">Module or Operators</span>
        </div>
        <div class="legend-item">
          <tf-graph-icon type="OP" height="16"></tf-graph-icon>
          <span class="legend-item-value">Unexpanded Module or Operators</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip fit-to-visible-bounds animation-delay="0" position="right" offset="0">
              <div class="custom-tooltip">
                Unexpandable Node: It can be an Api, operator or module. It cannot be expanded because it has no
                subnodes
              </div>
            </paper-tooltip>
          </div>
        </div>
        <div class="legend-item">
          <tf-graph-icon type="API_LIST" height="16"></tf-graph-icon>
          <span class="legend-item-value">Api List</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip animation-delay="0" position="right" offset="0">
              <div class="custom-tooltip">Apis between modules</div>
            </paper-tooltip>
          </div>
        </div>
        <div class="legend-item">
          <tf-graph-icon type="MULTI_COLLECTION" height="16"></tf-graph-icon>
          <span class="legend-item-value">Multi Collection</span>
          <div class="legend-item-value legend-clarifier">
            <paper-tooltip animation-delay="0" position="right" offset="0">
              <div class="custom-tooltip">Fusion node Collection</div>
            </paper-tooltip>
          </div>
        </div>
      </div>
    `;
  }
}
