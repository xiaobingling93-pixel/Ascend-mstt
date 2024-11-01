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

import { makeStyles } from '@material-ui/core/styles';
import * as React from 'react';
import { StepedGraph } from '../../api';
import { useResizeEventDependency } from '../../utils/resize';
import * as echarts from 'echarts';

interface IProps {
  graph: StepedGraph;
  height?: number;
  hAxisTitle?: string;
  vAxisTitle?: string;
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: Pick<IProps, 'height'>) => props.height,
  },
}));

export const SteppedAreaChart: React.FC<IProps> = (props) => {
  const { graph, height = 400, hAxisTitle, vAxisTitle } = props;
  const classes = useStyles({ height });
  const graphRef = React.useRef<HTMLDivElement>(null);
  const [resizeEventDependency] = useResizeEventDependency();

  React.useLayoutEffect(() => {
    const element = graphRef.current;
    if (!element) {return;}

    const chart = echarts.init(element);
    const dataSource: Array<Array<number | string>> = [];
    dataSource.push(graph.columns);
    graph.rows.forEach((row) => {
      dataSource.push(row.map((item) => item.value));
    });
    const options: echarts.EChartsOption = {
      title: {
        text: graph.title,
      },
      legend: {
        bottom: 0,
      },
      xAxis: {
        type: 'category',
        name: hAxisTitle,
        axisLabel: {
          interval: 0,
        },
      },
      yAxis: {
        type: 'value',
        name: vAxisTitle,
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          return (
            graph.rows[params.dataIndex][params.seriesIndex + 1]?.tooltip || ''
          );
        },
      },
      dataset: {
        source: dataSource,
      },
      series: Array(graph.columns.length - 1).fill({
        type: 'bar',
        stack: 'samesign',
      }),
    };

    if (options) {
      chart.setOption(options, true);
    };

    return () => {
      chart.dispose();
    };
  }, [graph, height, resizeEventDependency]);

  return (
    <div className={classes.root}>
      <div ref={graphRef} style={{ height }}></div>
    </div>
  );
};
