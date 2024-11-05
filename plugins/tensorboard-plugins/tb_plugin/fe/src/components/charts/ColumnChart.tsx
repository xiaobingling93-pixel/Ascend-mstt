/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------
 * Copyright (c) 2023, Huawei Technologies.
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
 *
 * Modifications: Offer offline supporting.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';
import { useResizeEventDependency } from '../../utils/resize';
import * as echarts from 'echarts';

interface IProps {
  title?: string;
  units?: string;
  colors?: Array<string>;
  chartData: ColumnChartData;
}

export interface ColumnChartData {
  legends: Array<string>;
  barLabels: Array<string>;
  barHeights: Array<Array<number>>;
}

export const ColumnChart: React.FC<IProps> = (props) => {
  const { title, units, colors, chartData } = props;
  const { legends, barLabels, barHeights } = chartData;
  const graphRef = React.useRef<HTMLDivElement>(null);
  const [resizeEventDependency] = useResizeEventDependency();

  const getAngleByDataLength = (data: number) => {
    if (data < 10) {
      return 0;
    } else {
      // 数量越大越趋近于旋转90度
      return 90 * (1 - (10 / data));
    }
  };

  React.useLayoutEffect(() => {
    const element = graphRef.current;
    if (!element) {return;}

    const chart = echarts.init(element);
    const dataSource: Array<Array<number | string>> = [];
    dataSource.push(['worker', ...legends]);
    barHeights.forEach((item, index) => {
      barLabels[index] !== undefined &&
        dataSource.push([barLabels[index], ...item]);
    });
    const options: echarts.EChartsOption = {
      title: {
        text: title,
      },
      legend: {
        bottom: 0,
      },
      xAxis: {
        type: 'category',
        axisLabel: {
          interval: 0,
          rotate: getAngleByDataLength(barLabels.length),
          formatter: (name: string) => {
            const index = name.indexOf('@');
            const processedName = index > -1 ? name.slice(index + 1) : name; // 使用新变量处理
            return processedName.length > 16 ? `${processedName.slice(0, 14)}...` : processedName;
          },
        },
      },
      yAxis: {
        type: 'value',
        name: units,
        nameTextStyle: {
          fontSize: 16,
        },
      },
      tooltip: {
        trigger: 'item',
      },
      dataset: {
        source: dataSource,
      },
      series: Array(legends.length).fill({
        type: 'bar',
        stack: 'samesign',
      }),
    };
    if (colors) {
      options.color = colors.slice(0, barLabels.length);
    }

    options && chart.setOption(options, true);
    return () => {
      chart.dispose();
    };
  }, [title, chartData, resizeEventDependency]);

  return <div ref={graphRef} style={{ height: '500px' }}></div>;
};
