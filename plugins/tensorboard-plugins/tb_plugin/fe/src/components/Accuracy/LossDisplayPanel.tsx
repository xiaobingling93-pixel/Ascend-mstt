/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------
 * Copyright (c) 2023, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the 'License')
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';
import { useState, useLayoutEffect, useRef } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { FileInfo } from './entity';
import * as echarts from 'echarts';
import { Table } from 'antd';
import { ColumnsType } from 'antd/es/table';

interface IProps {
  fileList: FileInfo[];
}

const useStyles = makeStyles(() => ({
  root: {
    padding: 24,
    width: '100%',
    display: 'flex',
    flexDirection: 'column',
    height: '50%',
  },
  title: {
    height: 24,
    lineHeight: '24px',
    fontFamily: 'sans-serif',
    fontSize: 16,
    fontWeight: 700,
  },
  content: {
    flex: 1,
    display: 'flex',
  },
  lossChart: {
    height: '100%',
    flex: 1,
  },
  lossTable: {
    height: '100%',
    width: '32%',
  },
  tableHeader: {
    display: 'inline-block',
    width: 134,
    position: 'absolute',
    top: '50%',
    transform: 'translateY(-50%)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
}));

export const LossDisplayPanel: React.FC<IProps> = (props) => {
  const { fileList } = props;
  const classes = useStyles();
  const chartRef = useRef<HTMLDivElement>(null);
  const [pageSize, setPageSize] = useState(20);
  const totalRef = useRef<number>(0);

  const getColumns = (): ColumnsType<any> => {
    const columns: ColumnsType<any> = [
      {
        title: 'Iteration',
        key: 'iter',
        dataIndex: 'iter',
        width: '20%',
        fixed: 'left',
      },
    ];
    fileList.forEach((item, index) => {
      columns.push({
        title: () => (
          <div className={classes.tableHeader} title={item.fileName}>
            {item.fileName}
          </div>
        ),
        key: index,
        dataIndex: item.fileName,
        width: 150,
      });
    });
    return columns;
  };

  const getTableData = (): readonly any[] => {
    const dataSource: any[] = [];
    let allIters: number[] = [];
    fileList.forEach((file) => {
      allIters = allIters.concat(file.iters);
    });
    const uniqueIter = new Set(allIters.sort((a, b) => a - b));
    uniqueIter.forEach((iter, index) => {
      const fileLosses: { [fileName: string]: number | string } = {};
      fileList.forEach((file) => {
        fileLosses[file.fileName] = file.iterLosses[iter] ?? 'NA';
      });
      dataSource.push({
        key: `${iter}_${index}`,
        iter,
        ...fileLosses,
      });
    });
    totalRef.current = dataSource.length;
    return dataSource;
  };

  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size);
  };

  useLayoutEffect(() => {
    const element = chartRef.current;
    if (!element) {
      return undefined;
    }
    const echart = echarts.init(element);
    const dataset: echarts.DatasetComponentOption[] = [];
    const series: echarts.SeriesOption[] = [];
    fileList.forEach((item, index) => {
      dataset.push({
        source: item.losses,
      });
      series.push({
        type: 'line',
        name: item.fileName,
        datasetIndex: index,
        symbol: 'none',
      });
    });
    let option: echarts.EChartsOption = {
      title: {
        text: 'Loss Chart',
        textStyle: {
          fontSize: 12,
          color: '#000',
        },
      },
      tooltip: {
        trigger: 'axis',
        confine: true,
        axisPointer: {
          label: {
            precision: 0,
          },
        },
      },
      legend: {
        type: 'scroll',
        bottom: 0,
        tooltip: {
          show: true,
        },
        formatter: (name) => {
          // Show ellipsis and set tooltip for legends with too long name
          return name.length > 50 ? `${name.slice(0, 48)}...` : name;
        },
      },
      xAxis: {
        type: 'value',
        boundaryGap: false,
        name: 'Iteration',
        minInterval: 1,
      },
      yAxis: {
        type: 'value',
        name: 'Loss',
        scale: true,
      },
      dataZoom: {
        type: 'inside',
      },
      dataset,
      series,
    };

    if (option) {
      echart.setOption(option, true);
    }

    return () => {
      echart.dispose();
    };
  }, [fileList]);

  return (
    <div className={classes.root}>
      <div className={classes.title}>Loss Data</div>
      <div className={classes.content}>
        <div ref={chartRef} className={classes.lossChart}></div>
        <Table
          className={classes.lossTable}
          columns={getColumns()}
          dataSource={getTableData()}
          size='small'
          scroll={{
            x: 150 * fileList.length + 100,
            y:
              fileList.length < 2
                ? 'calc(100vh - 240px)'
                : 'calc(50vh - 185px)',
          }}
          pagination={{
            pageSize,
            pageSizeOptions: ['10', '20', '30', '50', '100'],
            total: totalRef.current,
            showTotal: (total) => `Total ${total} items`,
            onShowSizeChange,
          }}
        />
      </div>
    </div>
  );
};
