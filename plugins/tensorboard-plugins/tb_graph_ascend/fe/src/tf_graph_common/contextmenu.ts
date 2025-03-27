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
import { TfGraphScene } from './tf-graph-scene';
import * as tf_graph_parser from '../tf_graph_common/parser';
import { getColorByPrecisionIndex } from './node';
import { setRankJump } from '../tf_graph/tf-graph';
import { BENCH_PREFIX, NPU_PREFIX, EXPAND_NODE, DATA_SEND, DATA_RECEIVE, DATA_SEND_RECEIVE } from './common';
import { safeJSONParse } from '../utils';

export interface TitleFunction {
  (data: any): string;
}
/** Function that takes action based on item clicked in the context menu. */
export interface ActionFunction {
  (data: any): string;
}
/**
 * The interface for an item in the context menu
 */
export interface ContextMenuItem {
  title: TitleFunction;
  action: ActionFunction;
}
/**
 * Returns the top and left distance of the scene element from the top left
 * corner of the screen.
 */
function getOffset(sceneElement): { left: number; top: number } {
  let leftDistance = 0;
  let topDistance = 0;
  let currentElement = sceneElement;
  while (currentElement && currentElement.offsetLeft >= 0 && currentElement.offsetTop >= 0) {
    leftDistance += currentElement.offsetLeft - currentElement.scrollLeft;
    topDistance += currentElement.offsetTop - currentElement.scrollTop;
    currentElement = currentElement.offsetParent;
  }
  return {
    left: leftDistance,
    top: topDistance,
  };
}
/**
 * Returns the event listener, which can be used as an argument for the d3
 * selection.on function. Renders the context menu that is to be displayed
 * in response to the event.
 */
export function getMenu(sceneElement: TfGraphScene, nodeData): () => Promise<void> {
  let selectedNode = '';
  const menuNode = sceneElement.getContextMenu();
  const menuSelection = d3.select(sceneElement.getContextMenu());
  // Function called to populate the context menu.
  return async function (): Promise<void> {
    // Position and display the menu.
    let event = <MouseEvent>d3.event;
    const sceneOffset = getOffset(sceneElement);
    menuSelection
      .style('display', 'block')
      .style('left', `${event.clientX - sceneOffset.left + 1}px`)
      .style('top', `${event.clientY - sceneOffset.top + 1}px`);
    // Stop the event from propagating further.
    event.preventDefault();
    event.stopPropagation();
    function maybeCloseMenu(closeEvent?: any): void {
      if (closeEvent?.composedPath().includes(menuNode)) {
        return;
      }
      menuSelection.style('display', 'none');
      document.body.removeEventListener('mousedown', maybeCloseMenu, {
        capture: true,
      });
    }
    // Dismiss and remove the click listener as soon as there is a mousedown
    // on the document. We use capture listener so no component can stop
    // context menu from dismissing due to stopped propagation.
    document.body.addEventListener('mousedown', maybeCloseMenu, {
      capture: true,
    });
    // Add provided items to the context menu.
    menuSelection.text('');
    const { name } = nodeData.node;
    let nodePrefix = name.substring(0, 4);
    let side = '';
    let nodeName = name;
    if (nodePrefix === NPU_PREFIX) {
      side = 'NPU';
      nodeName = name.substring(4);
    } else if (nodePrefix === BENCH_PREFIX) {
      side = 'Bench';
      nodeName = name.substring(4);
    } else {
      nodePrefix = '';
    }

    // 设置 URL 参数
    const params = new URLSearchParams();
    params.set('side', side);
    params.set('node', nodeName);

    // 获取侧路径并解析
    const sidePath = `rank?${String(params)}`;
    const sideStr = await tf_graph_parser.fetchPbTxt(sidePath);
    const decodedStr = new TextDecoder().decode(sideStr);
    const decodedObj = safeJSONParse(decodedStr);
    if (decodedObj === null) {
      console.error('Error: loading contextmenu failed please check data or getMenu function.');
      return;
    }
    // 构建菜单选项
    let communicationsType = decodedObj.communications_type;
    const menuOptions = [{ text: EXPAND_NODE, action: 'expand' }];
    if (communicationsType || decodedObj[0]?.communications_type) {
      // 定义一个函数来生成 titleText
      const getTitleText = (type: string): string => {
        if (type === 'send') {
          return DATA_SEND;
        }
        if (type === 'receive') {
          return DATA_RECEIVE;
        }
        return DATA_SEND_RECEIVE;
      };

      // 如果 communicationsType 存在，直接处理
      if (communicationsType) {
        const titleText = getTitleText(communicationsType);
        menuOptions.push({ text: titleText, action: 'rank' });
      }
      // 如果 decodedObj 中存在 communications_type，遍历处理
      else if (decodedObj[0].communications_type) {
        decodedObj.forEach((item) => {
          const titleText = getTitleText(item.communications_type);
          menuOptions.push({ text: titleText, action: 'rank' });
        });
      }
    }
    let list = menuSelection.append('ul');
    list
      .selectAll('li')
      .data(menuOptions)
      .enter()
      .append('li')
      .on('click', (d, i) => {
        if (d.action === 'expand') {
          sceneElement.fire('parent-node-toggle-expand', { nodeData });
        }
        maybeCloseMenu();
      })
      .on('mouseover', (d, i, nodes) => {
        if (d.action === 'rank') {
          const parentLi = d3.select(nodes[i]);
          // 第一项是expand，rank一定从第二项开始，所以是i-1
          const id = decodedObj[i - 1]?.communications_type || decodedObj.communications_type;
          // 检查是否已经有子菜单，防止重复添加
          if (!parentLi.select(`#submenu-${id}`).empty()) {
            return;
          }
          // 动态生成子菜单
          let nodeInfo: Record<string, [number, string]>;
          if (decodedObj[0]?.communications_type) {
            nodeInfo = decodedObj[i - 1]?.nodes_info ?? {}; // 如果是 undefined，使用空对象
          } else {
            nodeInfo = decodedObj.nodes_info ?? {}; // 如果是 undefined，使用空对象
          }
          const subMenuOptions: { text: string; action: string; color: string }[] = []; // 定义 subMenuOptions 数组的类型
          for (const [key, value] of Object.entries(nodeInfo)) {
            const rank = `rank${key}`;
            const screen = getColorByPrecisionIndex(String(value[0]));
            const menuName = value[1];
            subMenuOptions.push({ text: rank, action: menuName, color: screen });
          }
          // 定义一个函数来设置样式，减少重复代码
          const setStyle = (element, styles): void => {
            Object.keys(styles).forEach((key) => {
              element.style(key, styles[key]);
            });
          };

          // 常用样式
          const submenuStyles1 = {
            position: 'absolute',
            left: '100%',
            top: '27px',
            background: '#e2e2e2',
            color: 'black',
          };

          const submenuStyles2 = {
            ...submenuStyles1,
            top: '54px', // 第二个项
          };

          // 创建第三个样式对象，只修改 top 属性
          const submenuStyles3 = {
            ...submenuStyles1,
            top: '81px', // 第三个项
          };

          // 鼠标悬停时的样式
          const hoverStyles = {
            border: '1px solid #000',
            color: 'black',
          };

          // 鼠标离开时的样式
          const normalStyles = {
            border: '1px solid rgb(199, 199, 199)',
          };

          const submenuStyles = [submenuStyles1, submenuStyles2, submenuStyles3];

          // 创建子菜单
          const submenu = parentLi
            .append('ul')
            .attr('class', 'submenu')
            .attr('id', `submenu-${id}`) // 动态生成唯一的 id
            .call(setStyle, submenuStyles[i - 1]);

          // 创建子菜单项
          submenu
            .selectAll('li')
            .data(subMenuOptions)
            .enter()
            .append('li')
            .text((dText) => dText.text)
            .style('background-color', (dColor) => dColor.color)
            .call(setStyle, normalStyles)
            .on('mouseover', function () {
              d3.select(this).call(setStyle, hoverStyles);
            })
            .on('mouseout', function () {
              d3.select(this).call(setStyle, normalStyles);
            })
            .on('click', (subD) => {
              // 添加子菜单点击逻辑
              selectedNode = `${nodePrefix}${subD.action}`;
              setRankJump(selectedNode);
              sceneElement.fire('contextMenuTag-changed', parseInt(subD.text.slice(4), 10));
              maybeCloseMenu();
            });
        }
      })
      .on('mouseleave', (d, i, nodes) => {
        if (d.action === 'rank') {
          d3.select(nodes[i]).select('.submenu').remove(); // 隐藏子菜单
        }
      })
      .text((d) => d.text);
  };
}
