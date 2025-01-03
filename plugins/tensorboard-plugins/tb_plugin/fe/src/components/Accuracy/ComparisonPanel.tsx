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
import { useState, useLayoutEffect, useRef, useEffect } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { FileInfo } from './entity';
import { Empty, Popover, Radio, RadioChangeEvent, Select, Table } from 'antd';
import { ColumnsType } from 'antd/es/table';
import * as echarts from 'echarts';
import { InfoCircleOutlined } from '@ant-design/icons';

interface IProps {
  fileList: FileInfo[];
}

interface ILineDataList {
  normal: number[][];
  absolute: number[][];
  relative: number[][];
}

const useStyles = makeStyles(() => ({
  root: {
    height: '50%',
    width: '100%',
    padding: '0 24px 24px',
    display: 'flex',
    flexDirection: 'column',
  },
  title: {
    height: 24,
    lineHeight: '24px',
    fontFamily: 'sans-serif',
    fontSize: 16,
    fontWeight: 700,
  },
  filter: {
    height: 40,
    lineHeight: '40px',
    '& .comparisonSelect': {
      margin: '0 8px',
    },
    '& .comparisonLabel': {
      marginRight: 8,
    },
    '& .comparisonBtn': {
      marginLeft: 20,
    },
    '& .infoLabel': {
      fontSize: 20,
    },
  },
  empty: {
    marginTop: 60,
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

export const ComparisonPanel: React.FC<IProps> = (props) => {
  const { fileList } = props;
  const classes = useStyles();
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [compareWay, setCompareWay] = useState<number>(0);
  const [pageSize, setPageSize] = useState(20);
  const [lineData, setLineData] = useState<ILineDataList | undefined>(undefined);
  const [tableData, setTableData] = useState<any[]>([]);
  const chartRef = useRef<HTMLDivElement>(null);

  const getColumns = (): ColumnsType<any> => {
    const columns: ColumnsType<any> = [
      {
        title: 'Iteration',
        key: 'iter',
        dataIndex: 'iter',
      },
    ];
    selectedFiles.forEach((item, index) => {
      columns.push({
        title: () => (
          <div className={classes.tableHeader} title={item}>
            {item}
          </div>
        ),
        key: index,
        dataIndex: item,
        width: '40%',
      });
    });
    return columns;
  };

  const compareFile = (fileNames: string[]): void => {
    if (fileNames.length < 2) {
      return;
    }
    const baseFile = fileList.find((item) => item.fileName === fileNames[0]);
    const expFile = fileList.find((item) => item.fileName === fileNames[1]);
    if (!!baseFile && !!expFile) {
      const commonIters: number[] = [];
      const lessIters = baseFile.iters.length <= expFile.iters.length ? baseFile.iters : expFile.iters;
      const moreIters = baseFile.iters.length > expFile.iters.length ? baseFile.iters : expFile.iters;
      lessIters.forEach((iter) => {
        if (moreIters.includes(iter)) {
          commonIters.push(iter);
        }
      });
      commonIters.sort((a, b) => a - b);
      const tempTableData: any[] = [];
      const tempChartData: ILineDataList = {
        normal: [],
        absolute: [],
        relative: [],
      };
      commonIters.forEach((iter, index) => {
        const baseLoss = baseFile.iterLosses[iter];
        const expLoss = expFile.iterLosses[iter];
        tempTableData.push({
          key: `${iter}_${index}`,
          iter,
          [baseFile.fileName]: baseLoss,
          [expFile.fileName]: expLoss,
        });
        tempChartData.normal.push([iter, expLoss - baseLoss]);
        tempChartData.absolute.push([iter, Math.abs(expLoss - baseLoss)]);
        tempChartData.relative.push([iter, baseLoss === 0 ? 0 : Math.abs(expLoss - baseLoss) / baseLoss]);
      });
      setTableData(tempTableData);
      setLineData(tempChartData);
    }
  };

  const onSelectChange = (value: string[]): void => {
    setSelectedFiles(value);
    compareFile(value);
  };

  const onRadioChange = (e: RadioChangeEvent): void => {
    setCompareWay(e.target.value);
  };

  const onShowSizeChange = (current: number, size: number): void => {
    setPageSize(size);
  };

  useLayoutEffect(() => {
    const element = chartRef.current;
    if (!element || !lineData) {
      return undefined;
    }
    const echart = echarts.init(element);
    let dataSource: number[][] = [];
    if (compareWay === 0) {
      dataSource = lineData.normal;
    } else if (compareWay === 1) {
      dataSource = lineData.absolute;
    } else {
      dataSource = lineData.relative;
    }
    const option: echarts.EChartsOption = {
      title: {
        text: 'Comparison Chart',
        textStyle: {
          fontSize: 12,
          color: '#000',
        },
      },
      legend: { bottom: 0 },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        name: 'Iteration',
      },
      yAxis: {
        type: 'value',
        name: 'Difference',
        scale: true,
      },
      tooltip: {
        trigger: 'axis',
        valueFormatter: (value) => (value as number).toFixed(6),
      },
      dataZoom: {
        type: 'inside',
      },
      dataset: {
        source: dataSource,
      },
      series: {
        type: 'line',
        name: 'Difference',
        symbol: 'none',
      },
    };

    if (option) {
      echart.setOption(option, true);
    }
    return () => {
      echart.dispose();
    };
  }, [compareWay, lineData]);

  useEffect(() => {
    const tempValue = selectedFiles.filter((item) => {
      return !!fileList.find((file) => file.fileName === item);
    });
    if (JSON.stringify(tempValue) === JSON.stringify(selectedFiles)) {
      compareFile(tempValue);
    }
    setSelectedFiles(tempValue);
  }, [fileList]);

  return (
    <div className={classes.root}>
      <div className={classes.title}>Comparison Data</div>
      <div className={classes.filter}>
        <span>Comparison objects: </span>
        <Select
          className='comparisonSelect'
          mode='multiple'
          allowClear
          value={selectedFiles}
          placeholder='Please select 2 comparison files'
          maxTagTextLength={12}
          onChange={onSelectChange}
          style={{ width: 300 }}
          options={fileList.map((file) => {
            return {
              value: file.fileName,
              label: file.fileName,
              disabled: !selectedFiles.includes(file.fileName) && selectedFiles.length > 1,
            };
          })}
        />
        <span className='comparisonLabel'>Comparison Setting: </span>
        <Radio.Group value={compareWay} onChange={onRadioChange}>
          <Radio value={0}>Comparison Normal</Radio>
          <Radio value={1}>Comparison Absolute</Radio>
          <Radio value={2}>Comparison Relative</Radio>
        </Radio.Group>
        <Popover
          content={
            <>
              <div>
                <b>Normal:</b> The real difference.
              </div>
              <div>
                <b>Absolute:</b> The absolute value of difference.
              </div>
              <div>
                <b>Relative:</b> The absolute value of difference divided by the loss value of the first file.
              </div>
            </>
          }
        >
          <InfoCircleOutlined className='infoLabel' />
        </Popover>
      </div>
      {selectedFiles.length < 2 ? (
        <Empty className={classes.empty} description='Select 2 comparison files in the drop-down list' />
      ) : (
        <div className={classes.content}>
          <div ref={chartRef} className={classes.lossChart}></div>
          <Table
            className={classes.lossTable}
            columns={getColumns()}
            dataSource={tableData}
            size='small'
            scroll={{
              y: 'calc(50vh - 185px)',
            }}
            pagination={{
              pageSize,
              pageSizeOptions: ['10', '20', '30', '50', '100'],
              total: tableData.length,
              showTotal: (total) => `Total ${total} items`,
              onShowSizeChange,
            }}
          />
        </div>
      )}
    </div>
  );
};
