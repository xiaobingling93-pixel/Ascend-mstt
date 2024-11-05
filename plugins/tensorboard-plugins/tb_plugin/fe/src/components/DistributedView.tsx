/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardHeader from '@material-ui/core/CardHeader';
import Grid from '@material-ui/core/Grid';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select, { SelectProps } from '@material-ui/core/Select';
import { makeStyles } from '@material-ui/core/styles';
import { Table } from 'antd';
import { ColumnsType } from 'antd/es/table';
import * as React from 'react';
import * as api from '../api';
import { DistributedGraph, GpuInfo, Graph } from '../api';
import { firstOrUndefined } from '../utils';
import { ColumnChart } from './charts/ColumnChart';
import { DataLoading } from './DataLoading';
import { GpuInfoTable } from './GpuInfoTable';
import { makeChartHeaderRenderer, useTooltipCommonStyles } from './helpers';
import {
  DistributedCommopsTableTooltip,
  DistributedGpuInfoTableTooltip,
  DistributedOverlapGraphTooltip,
  DistributedWaittimeGraphTooltip,
} from './TooltipDescriptions';

export interface IProps {
  run: string;
  worker: string;
  span: string;
}

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  verticalInput: {
    display: 'flex',
    alignItems: 'center',
  },
  inputWidth: {
    width: '4em',
  },
  inputWidthOverflow: {
    minWidth: '15em',
    whiteSpace: 'nowrap',
  },
  description: {
    marginLeft: theme.spacing(1),
  },
  table: {
    height: '100%',
    border: '1px solid #efefef',
    '& .ant-table-tbody > tr': {
      height: 20,
      fontSize: '10pt',
      '& > td': {
        padding: '0 8px!important',
      },
    },
  },
}));

export const DistributedView: React.FC<IProps> = (props) => {
  const tooltipCommonClasses = useTooltipCommonStyles();
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses),
    [tooltipCommonClasses]
  );

  let { run, worker, span } = props;
  const classes = useStyles();

  const [overlapGraph, setOverlapGraph] = React.useState<
    DistributedGraph | undefined
  >(undefined);
  const [waittimeGraph, setWaittimeGraph] = React.useState<
    DistributedGraph | undefined
  >(undefined);
  const [commopsTableData, setCommopsTableData] = React.useState<
    any | undefined
  >(undefined);
  const [gpuInfo, setGpuInfo] = React.useState<GpuInfo | undefined>(undefined);
  const [commopsTableTitle, setCommopsTableTitle] = React.useState('');
  const [commopsWorkers, setCommopsWorkers] = React.useState<string[]>([]);
  const [overlapSteps, setOverlapSteps] = React.useState<string[]>([]);
  const [waittimeSteps, setWaittimeSteps] = React.useState<string[]>([]);
  const [overlapStep, setOverlapStep] = React.useState('');
  const [waittimeStep, setWaittimeStep] = React.useState('');
  const [commopsWorker, setCommopsWorker] = React.useState('');
  const [columns, setColumns] = React.useState<ColumnsType<any>>([]);
  const [pageSize, setPageSize] = React.useState(30);

  React.useEffect(() => {
    if (waittimeSteps.includes('all')) {
      setWaittimeStep('all');
    } else {
      setWaittimeStep(firstOrUndefined(waittimeSteps) ?? '');
    }
  }, [waittimeSteps]);

  React.useEffect(() => {
    if (overlapSteps.includes('all')) {
      setOverlapStep('all');
    } else {
      setOverlapStep(firstOrUndefined(overlapSteps) ?? '');
    }
  }, [overlapSteps]);

  React.useEffect(() => {
    setCommopsWorker(firstOrUndefined(commopsWorkers) ?? '');
  }, [commopsWorkers]);

  React.useEffect(() => {
    api.defaultApi.distributedOverlapGet(run, 'All', span).then((resp) => {
      setOverlapGraph(resp);
      setOverlapSteps(Object.keys(resp.data));
    });
    api.defaultApi.distributedWaittimeGet(run, 'All', span).then((resp) => {
      setWaittimeGraph(resp);
      setWaittimeSteps(Object.keys(resp.data));
    });
    api.defaultApi.distributedCommopsGet(run, 'All', span).then((resp) => {
      setCommopsTableData(resp.data);
      setCommopsWorkers(Object.keys(resp.data));
      setCommopsTableTitle(resp.metadata.title);
    });
    api.defaultApi.distributedGpuinfoGet(run, 'All', span).then((resp) => {
      setGpuInfo(resp);
    });
  }, [run, worker, span]);

  const onCommopsWorkerChanged: SelectProps['onChange'] = (event) => {
    setCommopsWorker(event.target.value as string);
  };

  const onOverlapStepChanged: SelectProps['onChange'] = (event) => {
    setOverlapStep(event.target.value as string);
  };

  const onWaittimeStepChanged: SelectProps['onChange'] = (event) => {
    setWaittimeStep(event.target.value as string);
  };

  const getColumnChartData = (
    distributedGraph?: DistributedGraph,
    step?: string
  ) => {
    if (!distributedGraph || !step) {return undefined;}
    const barLabels = Object.keys(distributedGraph.data[step]);
    return {
      legends: distributedGraph.metadata.legends,
      barLabels,
      barHeights: barLabels.map((label) => distributedGraph.data[step][label]),
    };
  };
  const overlapData = React.useMemo(
    () => getColumnChartData(overlapGraph, overlapStep),
    [overlapGraph, overlapStep]
  );
  const waittimeData = React.useMemo(
    () => getColumnChartData(waittimeGraph, waittimeStep),
    [waittimeGraph, waittimeStep]
  );

  const getTableData = (tableData?: any, worker?: string) => {
    if (!tableData || !worker) {
      return [];
    }
    let dataInfo: api.Graph = tableData[worker];
    const stringCompare = (a: string, b: string) => a.localeCompare(b);
    const numberCompare = (a: number, b: number) => a - b;
    let column: any[] = dataInfo.columns.map((item) => {
      return {
        title: item.name,
        key: item.name,
        dataIndex: item.name,
        sorter:
          item.type == 'string'
            ? (a: any, b: any) => stringCompare(a[item.name], b[item.name])
            : (a: any, b: any) => numberCompare(a[item.name], b[item.name]),
      };
    });
    setColumns(column);
    return dataInfo.rows.map((row, index) => {
      if (row.length !== dataInfo.columns.length) {
        return null;
      }
      const dataRow: { [column: string]: number | string } = { key: index };
      dataInfo.columns.forEach((column, index) => {
        dataRow[column.name] = row[index] as string | number;
      });
      return dataRow;
    });
  };
  const commopsTable: any[] = React.useMemo(() => {
    return getTableData(commopsTableData, commopsWorker);
  }, [commopsTableData, commopsWorker]);

  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size);
  };

  return (
    <div className={classes.root}>
      <Card variant='outlined'>
        <CardHeader title='Distributed View' />
        <CardContent>
          <Grid container spacing={1}>
            {gpuInfo && (
              <Grid item sm={12}>
                <Card elevation={0}>
                  <CardHeader
                    title={chartHeaderRenderer(
                      gpuInfo.metadata.title,
                      DistributedGpuInfoTableTooltip
                    )}
                  />
                  <CardContent>
                    <GpuInfoTable gpuInfo={gpuInfo} />
                  </CardContent>
                </Card>
              </Grid>
            )}
            <Grid item sm={6}>
              <DataLoading value={overlapData}>
                {(chartData) => (
                  <Card elevation={0}>
                    <CardContent>
                      <Grid container spacing={1} alignItems='center'>
                        <Grid item>
                          <InputLabel id='overlap-step'>Step</InputLabel>
                        </Grid>
                        <Grid item>
                          <Select
                            labelId='overlap-step'
                            value={overlapStep}
                            onChange={onOverlapStepChanged}
                          >
                            {overlapSteps.map((step) => (
                              <MenuItem value={step}>{step}</MenuItem>
                            ))}
                          </Select>
                        </Grid>
                      </Grid>
                    </CardContent>
                    {overlapGraph?.metadata?.title && (
                      <CardHeader
                        title={chartHeaderRenderer(
                          overlapGraph?.metadata?.title,
                          DistributedOverlapGraphTooltip
                        )}
                      />
                    )}
                    <ColumnChart
                      units={overlapGraph?.metadata?.units}
                      chartData={chartData}
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>
            <Grid item sm={6}>
              <DataLoading value={waittimeData}>
                {(chartData) => (
                  <Card elevation={0}>
                    <CardContent>
                      <Grid container spacing={1} alignItems='center'>
                        <Grid item>
                          <InputLabel id='waittime-step'>Step</InputLabel>
                        </Grid>
                        <Grid item>
                          <Select
                            labelId='waittime-step'
                            value={waittimeStep}
                            onChange={onWaittimeStepChanged}
                          >
                            {waittimeSteps.map((step) => (
                              <MenuItem value={step}>{step}</MenuItem>
                            ))}
                          </Select>
                        </Grid>
                      </Grid>
                    </CardContent>
                    {waittimeGraph?.metadata?.title && (
                      <CardHeader
                        title={chartHeaderRenderer(
                          waittimeGraph?.metadata?.title,
                          DistributedWaittimeGraphTooltip
                        )}
                      />
                    )}
                    <ColumnChart
                      units={waittimeGraph?.metadata?.units}
                      colors={['#0099C6', '#DD4477', '#66AA00', '#B82E2E']}
                      chartData={chartData}
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>
            <Grid item sm={12}>
              <Grid container direction='column' spacing={0}>
                <CardHeader
                  title={chartHeaderRenderer(
                    commopsTableTitle,
                    DistributedCommopsTableTooltip
                  )}
                />
                <Card elevation={0}>
                  <CardContent>
                    <Grid item container spacing={1} alignItems='center'>
                      <Grid item>
                        <InputLabel id='table-worker'>Worker</InputLabel>
                      </Grid>
                      <Grid item>
                        <Select
                          labelId='table-worker'
                          value={commopsWorker}
                          onChange={onCommopsWorkerChanged}
                        >
                          {commopsWorkers.map((worker) => (
                            <MenuItem value={worker}>{worker}</MenuItem>
                          ))}
                        </Select>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
                <Grid item>
                  <Table
                    className={classes.table}
                    columns={columns}
                    size='small'
                    dataSource={commopsTable}
                    pagination={{
                      pageSize,
                      pageSizeOptions: ['20', '30', '50', '100'],
                      hideOnSinglePage: true,
                      onShowSizeChange,
                    }}
                  />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  );
};
