/*--------------------------------------------------------------------------------------------
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
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';
import { Graph, GraphAscend } from '../../api';
import { useResizeEventDependency } from '../../utils/resize';
import { binarySearch } from '../../utils/binarysearch';
import * as echarts from 'echarts';

interface IProps {
  graph: Graph | GraphAscend;
  height?: number;
  deviceTarget: string;
  tag: string;
  hAxisTitle?: string;
  vAxisTitle?: string;
  onSelectionChanged?: (start: number, end: number) => void;
  record?: any;
}

export const LineChart: React.FC<IProps> = (props) => {
  const {
    graph,
    height = 400,
    deviceTarget,
    tag,
    hAxisTitle,
    vAxisTitle,
    onSelectionChanged,
    record,
  } = props;
  const graphRef = React.useRef<HTMLDivElement>(null);
  const [resizeEventDependency] = useResizeEventDependency();
  const [chartObj, setChartObj] = React.useState<echarts.ECharts | undefined>();
  const selectedPoints = React.useRef<Array<number>>([]);

  React.useLayoutEffect(() => {
    const element = graphRef.current;
    if (!element) return;
    element.oncontextmenu = () => {
      return false;
    };

    let myChart = echarts.init(element);

    let option: echarts.EChartsOption = {
      title: {
        text: graph.title,
        textStyle: {
          fontSize: 16,
        },
      },
      tooltip: { trigger: 'axis' },
      legend: {
        type: 'scroll',
        bottom: 0,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        name: hAxisTitle,
      },
      yAxis: {
        type: 'value',
        name: vAxisTitle,
        scale: true,
      },
      toolbox: {
        feature: {
          dataZoom: {
            yAxisIndex: 'none',
          },
          restore: {},
        },
      },
    };

    if (deviceTarget === 'Ascend') {
      if (tag === 'Component') {
        const mixedTooltip: echarts.TooltipComponentOption = {
          trigger: 'axis',
          formatter: function (params: any) {
            var res = `${params[0].name} <br/>`;
            for (const item of params) {
              if (typeof item.value[item.encode.y[0]] === 'number') {
                res += `<span style="background: ${item.color}; 
                height:10px; 
                width: 10px; 
                border-radius: 50%;
                display: inline-block;
                margin-right:10px;">
                </span> 
                ${item.seriesName}: ${item.value[item.encode.y[0]]}<br/>`;
              }
            }
            return res;
          },
        };
        if (graph.columns.length <= 4) {
          let finalRows = graph.rows['PTA'] ?? graph.rows['GE'];
          if (graph.columns.length === 4) {
            const mergedAPPRows = graph.rows['APP'].map(
              (item: Array<number | null>) => {
                return [item[0], null, null, item[1]];
              }
            );
            finalRows = finalRows
              .concat(mergedAPPRows)
              .sort((a: any, b: any) => {
                return a[0] - b[0];
              });
          }
          option = {
            ...option,
            tooltip: mixedTooltip,
            dataset: {
              source: [
                graph.columns.map((column) => column.name),
                ...finalRows,
              ],
            },
            series: Array(graph.columns.length - 1).fill({
              type: 'line',
              select: {
                itemStyle: {
                  borderWidth: 5,
                  shadowBlur: 5,
                },
              },
              emphasis: {
                itemStyle: {
                  borderWidth: 5,
                  shadowBlur: 5,
                },
              },
              selectedMode: 'single',
            }),
          };
        } else if (graph.columns.length <= 6) {
          const datasetTitle = graph.columns.map((item) => item.name);
          let mergedGERows = graph.rows['GE'].map(
            (item: Array<number | null>) => {
              return [item[0], null, null, item[1], item[2]];
            }
          );
          if (graph.columns.length === 6) {
            const mergedAPPRows = graph.rows['APP'].map(
              (item: Array<number | null>) => {
                return [item[0], null, null, null, null, item[2]];
              }
            );
            mergedGERows = mergedGERows.concat(mergedAPPRows);
          }
          const finalRows = graph.rows['PTA']
            .concat(mergedGERows)
            .sort((a: any, b: any) => {
              return a[0] - b[0];
            });
          option = {
            ...option,
            tooltip: mixedTooltip,
            dataset: {
              source: [datasetTitle, ...finalRows],
            },
            series: Array(graph.columns.length - 1).fill({
              type: 'line',
              connectNulls: true,
              select: {
                itemStyle: {
                  borderWidth: 5,
                  shadowBlur: 5,
                },
              },
              emphasis: {
                itemStyle: {
                  borderWidth: 5,
                  shadowBlur: 5,
                },
              },
              selectedMode: 'single',
              datasetIndex: 0,
            }),
          };
        }
      } else {
        if (graph.columns.length === 3) {
          const datasetTitle1: Array<string> = [];
          const datasetTitle2: Array<string> = [];
          graph.columns.forEach((column, index) => {
            if (index === 0 || index < 2) {
              datasetTitle1.push(column.name);
            }
            if (index === 0 || index >= 2) {
              datasetTitle2.push(column.name);
            }
          });
          option = {
            ...option,
            dataset: [
              {
                source: [datasetTitle1, ...graph.rows['Allocated']],
              },
              {
                source: [datasetTitle2, ...graph.rows['Reserved']],
              },
            ],
            series: [
              {
                type: 'line',
                name: 'Allocated',
                emphasis: {
                  label: {
                    show: true,
                  },
                  itemStyle: {
                    borderWidth: 5,
                    shadowBlur: 5,
                  },
                },
                select: {
                  itemStyle: {
                    borderWidth: 5,
                    shadowBlur: 5,
                  },
                },
                datasetIndex: 0,
              },
              {
                type: 'line',
                name: 'Reserved',
                select: {
                  itemStyle: {
                    borderWidth: 5,
                    shadowBlur: 5,
                  },
                },
                emphasis: {
                  itemStyle: {
                    borderWidth: 5,
                    shadowBlur: 5,
                  },
                },
                selectedMode: 'single',
                datasetIndex: 1,
              },
            ],
          };
        }
      }
    } else {
      option = {
        ...option,
        dataset: {
          source: [graph.columns.map((column) => column.name), ...graph.rows],
        },
        series: [
          {
            type: 'line',
            name: 'Allocated',
            select: {
              itemStyle: {
                borderWidth: 5,
                shadowBlur: 5,
              },
            },
            emphasis: {
              itemStyle: {
                borderWidth: 5,
                shadowBlur: 5,
              },
            },
            selectedMode: 'single',
          },
          {
            type: 'line',
            name: 'Reserved',
            select: {
              itemStyle: {
                borderWidth: 5,
                shadowBlur: 5,
              },
            },
            emphasis: {
              itemStyle: {
                borderWidth: 5,
                shadowBlur: 5,
              },
            },
            selectedMode: 'single',
          },
        ],
      };
    }

    option && myChart.setOption(option, true);
    myChart.dispatchAction({
      type: 'takeGlobalCursor',
      key: 'dataZoomSelect',
      dataZoomSelectActive: true,
    });

    myChart.on('dataZoom', (param: any) => {
      if (onSelectionChanged) {
        onSelectionChanged(param.batch[0].startValue, param.batch[0].endValue);
      }
    });

    myChart.on('restore', () => {
      if (onSelectionChanged) {
        // Set startId greater than endId to query all memory events.
        onSelectionChanged(0, -1);
      }
    });

    myChart.on('click', (param) => {
      myChart.dispatchAction({
        type: 'unselect',
        seriesId: param.seriesId,
        dataIndex: selectedPoints.current,
      });
      myChart.dispatchAction({
        type: 'select',
        seriesId: param.seriesId,
        dataIndex: param.dataIndex,
      });

      selectedPoints.current = [param.dataIndex];
    });

    myChart.getZr().on('contextmenu', () => {
      myChart.dispatchAction({
        type: 'restore',
      });
      myChart.dispatchAction({
        type: 'takeGlobalCursor',
        key: 'dataZoomSelect',
        dataZoomSelectActive: true,
      });
    });

    setChartObj(myChart);
    return () => {
      myChart.dispose();
    };
  }, [graph, height, resizeEventDependency]);

  React.useEffect(() => {
    const compare_fn = (key: number, mid: Array<number>) => key - mid[0];
    if (chartObj && tag === 'Operator') {
      if (record) {
        let startId = -1;
        let endId = -1;
        if (deviceTarget === 'Ascend') {
          startId = binarySearch(
            graph.rows['Allocated'],
            record.col2,
            compare_fn
          );
          endId = binarySearch(
            graph.rows['Allocated'],
            record.col3,
            compare_fn
          );
        } else {
          startId = binarySearch(graph.rows, record.col2, compare_fn);
          endId = binarySearch(graph.rows, record.col3, compare_fn);
        }
        let selection = [];
        startId >= 0 && selection.push(startId);
        endId >= 0 && selection.push(endId);
        chartObj.dispatchAction({
          type: 'downplay',
          seriesName: 'Allocated',
          dataIndex: selectedPoints.current,
        });
        chartObj.dispatchAction({
          type: 'highlight',
          seriesName: 'Allocated',
          dataIndex: selection,
        });
        selectedPoints.current = selection;
      } else {
        chartObj.dispatchAction({
          type: 'downplay',
          seriesName: 'Allocated',
          dataIndex: selectedPoints.current,
        });
        selectedPoints.current = [];
      }
    }
  }, [graph, record, chartObj]);

  return <div ref={graphRef} style={{ height: '400px' }}></div>;
};
