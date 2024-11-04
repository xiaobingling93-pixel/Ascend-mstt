/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardHeader from '@material-ui/core/CardHeader';
import Grid from '@material-ui/core/Grid';
import { makeStyles } from '@material-ui/core/styles';
import { Table } from 'antd';
import { ColumnsType } from 'antd/es/table';
import * as React from 'react';
import * as api from '../api';
import { PieChart } from './charts/PieChart';
import { SteppedAreaChart } from './charts/SteppedAreaChart';
import { DataLoading } from './DataLoading';
import { makeChartHeaderRenderer, useTooltipCommonStyles } from './helpers';
import { TextListItem } from './TextListItem';
import { StepTimeBreakDownTooltip } from './TooltipDescriptions';
import {
  transformPerformanceIntoPie,
  transformPerformanceIntoTable,
} from './transform';

const topGraphHeight = 230;

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  pre: {
    '& ul': {
      margin: 0,
      paddingLeft: theme.spacing(3),
      ...theme.typography.body1,
    },
    '& li': {},
    '& a': {
      color: '#ffa726',
    },
    '& a:active': {
      color: '#ffa726',
    },
    '& p': {
      margin: 0,
      ...theme.typography.subtitle1,
      fontWeight: theme.typography.fontWeightBold,
    },
  },
  topGraph: {
    height: topGraphHeight + 40,
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

export interface IProps {
  run: string;
  worker: string;
  span: string;
}

export const Overview: React.FC<IProps> = (props) => {
  const { run, worker, span } = props;

  const [steps, setSteps] = React.useState<api.StepedGraph | undefined>(
    undefined
  );
  const [performances, setPerformances] = React.useState<api.Performance[]>([]);
  const [environments, setEnvironments] = React.useState<api.Environment[]>([]);
  const [gpuMetrics, setGpuMetrics] = React.useState<
    api.GpuMetrics | undefined
  >(undefined);
  const [recommendations, setRecommendations] = React.useState('');
  const [columns, setColumns] = React.useState<ColumnsType<any>>([]);

  const tableRows = React.useMemo(() => {
    let dataInfo: api.Graph = transformPerformanceIntoTable(performances);
    if (dataInfo.columns.length < 3) {
      return [];
    }
    const stringCompare = (a: string, b: string) => a.localeCompare(b);
    const numberCompare = (a: number, b: number) => a - b;
    let column: any[] = dataInfo.columns.map((item) => {
      return {
        title: item.name,
        key: item.name,
        dataIndex: item.name,
        sorter:
          item.type === 'string'
            ? (a: any, b: any) => stringCompare(a[item.name], b[item.name])
            : (a: any, b: any) => numberCompare(a[item.name], b[item.name]),
      };
    });
    setColumns(column);
    return dataInfo.rows.map((row, index) => {
      if (row.length < 3) {
        return null;
      }
      return {
        key: index,
        [dataInfo.columns[0].name]: row[0],
        [dataInfo.columns[1].name]: row[1],
        [dataInfo.columns[2].name]: row[2],
      };
    });
  }, [performances]);

  const synthesizedPieGraph = React.useMemo(() => {
    return transformPerformanceIntoPie(performances);
  }, [performances]);

  React.useEffect(() => {
    api.defaultApi.overviewGet(run, worker, span).then((resp) => {
      setPerformances(resp.performance);
      setEnvironments(resp.environments);
      setSteps(resp.steps);
      setRecommendations(resp.recommendations);
      setGpuMetrics(resp.gpu_metrics);
    });
  }, [run, worker, span]);

  const classes = useStyles();
  const tooltipCommonClasses = useTooltipCommonStyles();
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses, false),
    [tooltipCommonClasses]
  );

  const stepTimeBreakDownTitle = React.useMemo(
    () => chartHeaderRenderer('Step Time Breakdown', StepTimeBreakDownTooltip),
    [tooltipCommonClasses, chartHeaderRenderer]
  );

  const cardSizes = gpuMetrics
    ? ([2, 3, 7] as const)
    : ([4, undefined, 8] as const);

  return (
    <div className={classes.root}>
      <Grid container spacing={1}>
        <Grid container item spacing={1}>
          <Grid item sm={cardSizes[0]}>
            {React.useMemo(
              () => (
                <Card variant='outlined'>
                  <CardHeader title='Configuration' />
                  <CardContent className={classes.topGraph}>
                    {environments.map((environment) => (
                      <TextListItem
                        name={environment.title}
                        value={environment.value}
                      />
                    ))}
                  </CardContent>
                </Card>
              ),
              [environments]
            )}
          </Grid>
          {gpuMetrics && (
            <Grid item sm={cardSizes[1]}>
              <Card variant='outlined'>
                <CardHeader
                  title={chartHeaderRenderer('GPU Summary', gpuMetrics.tooltip)}
                />
                <CardContent
                  className={classes.topGraph}
                  style={{ overflow: 'auto' }}
                >
                  {gpuMetrics.data.map((metric) => (
                    <TextListItem
                      name={metric.title}
                      value={metric.value}
                      dangerouslyAllowHtml
                    />
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}
          <Grid item sm={cardSizes[2]}>
            <Card variant='outlined'>
              <CardHeader title='Execution Summary' />
              <CardContent>
                <Grid container spacing={1}>
                  <Grid item sm={6}>
                    <Table
                      className={classes.table}
                      columns={columns}
                      size='small'
                      dataSource={tableRows}
                      pagination={false}
                    />
                  </Grid>
                  <Grid item sm={5}>
                    <PieChart
                      graph={synthesizedPieGraph}
                      height={topGraphHeight}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container item>
          <Grid item sm={12}>
            <Card variant='outlined'>
              <CardHeader title={stepTimeBreakDownTitle} />
              <CardContent>
                <DataLoading value={steps}>
                  {(graph) => (
                    <SteppedAreaChart
                      graph={graph}
                      hAxisTitle='Step'
                      vAxisTitle={'Step Time (microseconds)'}
                    />
                  )}
                </DataLoading>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container item>
          <Grid item sm={12}>
            <Card variant='outlined'>
              <CardHeader title='Performance Recommendation' />
              <CardContent>
                <div className={classes.pre}>
                  <div
                    dangerouslySetInnerHTML={{
                      __html: recommendations || 'None',
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Grid>
    </div>
  );
};
