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

import '@vaadin/button';
import '@vaadin/details';
import '@vaadin/combo-box';

import { isEmpty } from 'lodash';
import { Notification } from '@vaadin/notification';
import { PolymerElement, html } from '@polymer/polymer';
import { customElement, property } from '@polymer/decorators';
import '@vaadin/progress-bar';
@customElement('tf-search-combox')
class Legend extends PolymerElement {
  // 定义模板
  static readonly template = html`
    <style>
      .search-arrow {
        font-size: 10px;
        margin-top: 27px;
        cursor: pointer;
        color: rgb(87, 86, 86);
        background: rgb(238, 238, 238);
        border: 0.5px solid var(--paper-input-container-color, var(--secondary-text-color));
        padding: 2px 2px;
        height: 30px;
        width: 22px;
      }
      .search-arrow:hover {
        background: rgb(201, 200, 199);
      }
      .container-search {
        display: flex;
        align-items: center;
        padding-right: 8px;
      }
      vaadin-combo-box {
        width: 100%;
      }
      vaadin-combo-box::part(input-field) {
        height: 30px;
        border: 1px solid var(--paper-input-container-color, var(--secondary-text-color));
        background-color: white;
        font-size: 14px;
        border-radius: 0;
      }
      vaadin-combo-box::part(toggle-button) {
        font-size: 14px;
      }
      .search-combox {
        display: flex;
        align-items: center;
        width: 100%;
      }
    </style>
    <div class="search-combox">
      <vaadin-combo-box
        label="[[label]]"
        items="[[items]]"
        value="{{selectedValue}}"
        on-change="_onChange"
      ></vaadin-combo-box>
      <vaadin-icon
        title="搜索上一项"
        icon="vaadin:arrow-up"
        class="search-arrow"
        on-click="_selectPrevious"
      ></vaadin-icon>
      <vaadin-icon
        title="搜索下一项"
        icon="vaadin:arrow-down"
        class="search-arrow"
        on-click="_selectNext"
      ></vaadin-icon>
    </div>
  `;

  @property({ type: Object })
  onSelectChange!: () => void;

  @property({ type: String, notify: true })
  selectedValue: string = '';

  @property({ type: Array })
  items: string[] = [];

  @property({ type: Boolean })
  isCompareGraph: boolean = true;

  _onChange(): void {
    this.onSelectChange();
  }

  // 选择列表中的下一个节点
  _selectNext(): void {
    if (!this.isCompareGraph) {
      Notification.show('提示：单图节点不支持匹配', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }

    if (isEmpty(this.items)) {
      Notification.show('提示：列表为空', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (isEmpty(this.selectedValue)) {
      this.set('selectedValue', this.items[0]);
      this.onSelectChange();
      return;
    }
    const index = this.items.indexOf(this.selectedValue);
    if (index + 1 >= this.items.length) {
      Notification.show('提示：已到达列表底部', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    } else {
      this.set('selectedValue', this.items[index + 1]);
    }
    this.onSelectChange();
  }

  // 选择列表中的上一个节点
  _selectPrevious(): void {
    if (!this.isCompareGraph) {
      Notification.show('提示：单图节点不支持匹配', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }

    if (isEmpty(this.items)) {
      Notification.show('提示：列表为空', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    }
    if (isEmpty(this.selectedValue)) {
      this.set('selectedValue', this.items[0]);
      this.onSelectChange();
      return;
    }
    const index = this.items.indexOf(this.selectedValue);
    if (index - 1 < 0) {
      Notification.show('提示：已到达列表顶部', {
        position: 'middle',
        duration: 2000,
        theme: 'contrast',
      });
      return;
    } else {
      this.set('selectedValue', this.items[index - 1]);
    }
    this.onSelectChange();
  }
}
