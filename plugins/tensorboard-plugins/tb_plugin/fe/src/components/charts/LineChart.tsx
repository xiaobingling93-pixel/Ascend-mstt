/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles';
import * as React from 'react';
import { Graph, GraphAscend } from '../../api';
import { useResizeEventDependency } from '../../utils/resize';
import { binarySearch } from '../../utils/binarysearch';

interface IProps {
  graph: Graph | GraphAscend;
  height?: number;
  deviceTarget: string;
  tag: string;
  hAxisTitle?: string;
  vAxisTitle?: string;
  explorerOptions?: object;
  onSelectionChanged?: (start: number, end: number) => void;
  record?: any;
}

const useStyles = makeStyles(() => ({
  root: {
    height: (props: Pick<IProps, 'height'>) => props.height,
  },
}));

export const LineChart: React.FC<IProps> = (props) => {
  const {
    graph,
    height = 400,
    deviceTarget,
    tag,
    hAxisTitle,
    vAxisTitle,
    onSelectionChanged,
    explorerOptions,
    record,
  } = props;
  const classes = useStyles({ height });
  const graphRef = React.useRef<HTMLDivElement>(null);
  const [resizeEventDependency] = useResizeEventDependency();
  const [chartObj, setChartObj] = React.useState<any | undefined>();

  React.useLayoutEffect(() => {
    const element = graphRef.current;
    if (!element) {return;}

    const options = {
      title: graph.title,
      isStacked: true,
      height,
      legend: { position: 'bottom' },
      tooltip: { isHtml: true },
      hAxis: {
        title: hAxisTitle,
      },
      vAxis: {
        title: vAxisTitle,
      },
      explorer: explorerOptions,
    };

    const chart = new google.visualization.LineChart(element);

    // Disable selection of single point
    google.visualization.events.addListener(chart, 'select', function () {
      chart.setSelection();
    });

    google.visualization.events.addListener(chart, 'ready', function () {
      let zoomLast = getCoords();
      let observer = new MutationObserver(function () {
        var zoomCurrent = getCoords();
        if (JSON.stringify(zoomLast) !== JSON.stringify(zoomCurrent)) {
          zoomLast = getCoords();
          if (onSelectionChanged) {
            onSelectionChanged(zoomLast.x_min, zoomLast.x_max);
          }
        }
      });
      if (graphRef.current) {
        observer.observe(graphRef.current, {
          childList: true,
          subtree: true,
        });
      }
    });

    function getCoords() {
      let chartLayout = chart.getChartLayoutInterface();
      let chartBounds = chartLayout.getChartAreaBoundingBox();

      return {
        x_min: chartLayout.getHAxisValue(chartBounds.left),
        x_max: chartLayout.getHAxisValue(chartBounds.width + chartBounds.left),
      };
    }

    if (deviceTarget === 'Ascend') {
      let data = new google.visualization.DataTable();
      if (tag === 'Component') {
        if (graph.columns.length === 3) {
          graph.columns.forEach((column) => {
            data.addColumn({
              type: column.type,
              label: column.name,
              role: column.role,
              p: column.p,
            });
          });
          data.addRows(graph.rows.PTA ?? graph.rows.GE);
        } else if (graph.columns.length === 5) {
          const data2 = new google.visualization.DataTable();
          graph.columns.forEach((column, index) => {
            if (index === 0 || index < 3) {
              data.addColumn({
                type: column.type,
                label: column.name,
                role: column.role,
                p: column.p,
              });
            }
            if (index === 0 || index >= 3) {
              data2.addColumn({
                type: column.type,
                label: column.name,
                role: column.role,
                p: column.p,
              });
            }
          });
          data.addRows(graph.rows.PTA);
          data2.addRows(graph.rows.GE);
          data = google.visualization.data.join(
            data,
            data2,
            'full',
            [[0, 0]],
            [1, 2],
            [1, 2]
          );
        }
      } else {
        if (graph.columns.length === 2) {
          graph.columns.forEach((column) => {
            data.addColumn({
              type: column.type,
              label: column.name,
              role: column.role,
              p: column.p,
            });
          });
          data.addRows(graph.rows.Allocated ?? graph.rows.Reserved);
        } else if (graph.columns.length === 3) {
          const data2 = new google.visualization.DataTable();
          graph.columns.forEach((column, index) => {
            if (index === 0 || index < 2) {
              data.addColumn({
                type: column.type,
                label: column.name,
                role: column.role,
                p: column.p,
              });
            }
            if (index === 0 || index >= 2) {
              data2.addColumn({
                type: column.type,
                label: column.name,
                role: column.role,
                p: column.p,
              });
            }
          });
          data.addRows(graph.rows.Allocated);
          data2.addRows(graph.rows.Reserved);
          data = google.visualization.data.join(
            data,
            data2,
            'full',
            [[0, 0]],
            [1],
            [1]
          );
        }
      }

      chart.draw(data, options);
    } else {
      const data = new google.visualization.DataTable();
      graph.columns.forEach((column) => {
        data.addColumn({
          type: column.type,
          label: column.name,
          role: column.role,
          p: column.p,
        });
      });
      data.addRows(graph.rows);
      chart.draw(data, options);
    }

    setChartObj(chart);
  }, [graph, height, resizeEventDependency]);

  React.useEffect(() => {
    const compare_fn = (key: number, mid: Array<number>) =>
      key - parseFloat(mid[0].toFixed(2));
    if (chartObj && tag === 'Operator') {
      if (record) {
        if (deviceTarget === 'Ascend') {
          let startId = binarySearch(
            graph.rows.Allocated,
            record.col2,
            compare_fn
          );
          let endId = binarySearch(
            graph.rows.Allocated,
            record.col3,
            compare_fn
          );
          let selection = [];
          if (startId >= 0) {selection.push({ row: startId, column: 1 });}
          if (endId >= 0) {selection.push({ row: endId, column: 1 });}
          chartObj.setSelection(selection);
        } else {
          let startId = binarySearch(graph.rows, record.col2, compare_fn);
          let endId = binarySearch(graph.rows, record.col3, compare_fn);
          let selection = [];
          if (startId >= 0) {selection.push({ row: startId, column: 1 });}
          if (endId >= 0) {selection.push({ row: endId, column: 1 });}
          chartObj.setSelection(selection);
        }
      } else {
        chartObj.setSelection();
      }
    }
  }, [graph, record, chartObj]);

  return (
    <div className={classes.root}>
      <div ref={graphRef}></div>
    </div>
  );
};
