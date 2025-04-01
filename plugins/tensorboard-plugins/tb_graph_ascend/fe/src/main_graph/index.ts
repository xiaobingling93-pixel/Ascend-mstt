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
import { customElement, property, query } from '@polymer/decorators';
import '@vaadin/grid'; // 引入新的 Vaadin Grid 组件
import '@vaadin/tooltip';
import * as d3 from 'd3';


export enum NodeType {
    MODULE = 0,
    OPERATOR = 1,
    MULTI_COLLECTION = 8,
    API_LIST = 9,
}

@customElement('main-graph')
class MainGraph extends PolymerElement {

    static readonly template = html`
       <style>
        :host {
          display: block;
          width: 100%;
          height: 500px;
        }
        svg {
          width: 100%;
          height: 100%;
        }
        rect {
    
        }
        text {
          font-family: Arial, sans-serif;
          font-size: 8px;
          fill: black;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
      </style>
      <button on-click="updateData">更新数据</button>
      <svg width="100%" height="100%" id="graph" transform="translate(370,290) scale(2)"></svg>  
  `;

    @property({ type: Array })
    data = [
        {
            name: 'AddThree_0',
            width: 316,
            height: 124,
            color: 'rgb(255, 237, 190)',
            x: 138,
            y: 0,
            nodeType: 0,
            flod: false
        },
        {
            name: 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.tt.ee',
            width: 300,
            height: 15,
            color: 'rgb(199, 199, 199)',
            x: 150,
            y: 74.5,
            nodeType: 0,
            flod: true
        },
        {
            name: 'maxpoolMaxPool2.maxpoolpo.tt.ee',
            width: 168,
            height: 15,
            color: 'rgb(182, 199, 252',
            x: 200,
            y: 49.5,
            nodeType: 0,
            flod: true
        },
        {
            name: 'arg0_1_0',
            width: 40,
            height: 12,
            color: 'rgb(199, 199, 199)',
            x: 260,
            y: 26,
            nodeType: 0,
            flod: true
        },
        {
            name: 'output_0',
            width: 40,
            height: 12,
            color: 'rgb(199, 199, 199)',
            x: 260,
            y: 98,
            nodeType: 0,
            flod: true
        }
    ];

    override ready(): void {
        super.ready();
        this.renderDiagram(this.data);
    }

    renderDiagram(data) {
        if (!this.shadowRoot) return;
        const svg = d3.select(this.shadowRoot.querySelector('#graph'));
        // 绑定数据到 g 元素
        const groups = svg.selectAll('g').data(data, (d: any) => d.name);

        // 更新现有矩形和形状（带动画）
        groups.each(function (d: any) {
            const nodeElement = d3.select(this);

            // 更新矩形的属性
            nodeElement.selectAll('rect')
                .transition()
                .duration(500) // 动画持续时间
                .attr('x', d.x)
                .attr('y', d.y)
                .attr('width', d.width)
                .attr('height', d.type === NodeType.MODULE && !d.flod ? d.height : 15)
                .attr('fill', d.color)
                .attr('stroke', d.color)
                .attr('stroke-width', 1);

            // 更新文本位置
            nodeElement.selectAll('text')
                .transition()
                .duration(500)
                .attr('x', d.x + d.width / 2)
                .attr('y', d.y + (d.type === NodeType.MODULE ? d.height / 2 : 8));
        });

        // 添加新矩形和形状
        const newGroups = groups.enter()
            .append('g')
            .attr('name', (d: any) => d.name);

        newGroups.each(function (d: any) {
            const nodeElement = d3.select(this);
            if (d.nodeType === NodeType.MODULE) {
                if (!d.flod) {
                    nodeElement.append('rect')
                        .attr('x', d.x)
                        .attr('y', d.y)
                        .attr('width', 0) // 初始宽度为 0，用于动画
                        .attr('height', 0) // 初始高度为 0，用于动画
                        .attr('rx', '5')
                        .attr('ry', '5')
                        .attr('fill', 'white')
                        .attr('stroke', d.color)
                        .attr('stroke-width', 1)
                        .transition()
                        .duration(500)
                        .attr('width', d.width)
                        .attr('height', d.height);
                }
                nodeElement.append('rect')
                    .attr('x', d.x)
                    .attr('y', d.y + 1)
                    .attr('rx', '5')
                    .attr('ry', '5')
                    .attr('width', 0) // 初始宽度为 0，用于动画
                    .attr('height', 0) // 初始高度为 0，用于动画
                    .attr('fill', d.color)
                    .attr('stroke', d.color)
                    .attr('stroke-width', 1)
                    .transition()
                    .duration(500)
                    .attr('width', d.width)
                    .attr('height', 15);

                nodeElement.append('text')
                    .attr('x', d.x + d.width / 2)
                    .attr('y', d.y + 8)
                    .attr('dy', '0.35em')
                    .attr('text-anchor', 'middle')
                    .style('opacity', 0) // 初始透明度为 0
                    .text(d.name)
                    .transition() // 添加动画
                    .duration(1000) // 动画持续时间
                    .style('opacity', 1); // 目标透明度为 1
            }
            // 可以根据需要添加其他类型的节点（如 OPERATOR、API_LIST 等）
        });

        // 删除多余的矩形和形状（带动画）
        groups.exit()
            .selectAll('rect')
            .transition()
            .duration(500)
            .attr('height', 0) // 高度逐渐变为 0
            .remove();

        groups.exit()
            .selectAll('text')
            .transition()
            .duration(500)
            .style('opacity', 0) // 文本透明度逐渐变为 0
            .remove();

        groups.exit()
            .transition()
            .duration(500)
            .remove();
    }

    updateData() {
        // 随机生成新的数据
        // 模拟数据变化：随机更新高度
        this.data = this.generateData(10);
        console.log(this.data);
        this.renderDiagram(this.data);
    }

    generateData(count) {
        const data: any = [];
        const colors = ['red', 'blue', 'green', 'orange', 'purple']; // 颜色选项

        for (let i = 0; i < count; i++) {
            const name = `Item_${i}`; // 生成唯一的名称
            const width = Math.floor(Math.random() * 300 + 50); // 宽度范围：50 ~ 350
            const height = Math.floor(Math.random() * 100 + 10); // 高度范围：10 ~ 110
            const color = colors[Math.floor(Math.random() * colors.length)]; // 随机颜色
            const x = Math.floor(Math.random() * 500); // x 范围：0 ~ 500
            const y = Math.floor(Math.random() * 500); // y 范围：0 ~ 500

            data.push({
                name,
                width,
                height,
                color,
                x,
                y,
            });
        }

        return data;
    }
}
