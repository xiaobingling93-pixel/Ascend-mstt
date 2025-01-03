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
 * Modifications: Add visualization of PyTorch Ascend profiling.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardHeader from '@material-ui/core/CardHeader';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Grid from '@material-ui/core/Grid';
import GridList from '@material-ui/core/GridList';
import GridListTile from '@material-ui/core/GridListTile';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Radio from '@material-ui/core/Radio';
import RadioGroup, { RadioGroupProps } from '@material-ui/core/RadioGroup';
import Select, { SelectProps } from '@material-ui/core/Select';
import { makeStyles } from '@material-ui/core/styles';
import TextField, { StandardTextFieldProps, TextFieldProps } from '@material-ui/core/TextField';
import * as React from 'react';
import * as api from '../api';
import { OperationTableData, OperationTableDataInner, OperatorGraph } from '../api';
import { OperationGroupBy } from '../constants/groupBy';
import { useSearchDirectly } from '../utils/search';
import { topIsValid, UseTop, useTopN } from '../utils/top';
import { PieChart } from './charts/PieChart';
import { DataLoading } from './DataLoading';
import { makeChartHeaderRenderer, useTooltipCommonStyles } from './helpers';
import { OperationTable } from './tables/OperationTable';
import {
  deviceSelfTimeTooltip,
  deviceSelfTimeTooltipAscend,
  deviceTotalTimeTooltip,
  deviceTotalTimeTooltipAscend,
  hostSelfTimeTooltip,
  hostTotalTimeTooltip,
} from './TooltipDescriptions';

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
  full: {
    width: '100%',
  },
  description: {
    marginLeft: theme.spacing(1),
  },
}));

export interface IProps {
  run: string;
  worker: string;
  span: string;
  deviceTarget: string;
}

export const Operator: React.FC<IProps> = (props) => {
  const { run, worker, span, deviceTarget } = props;
  const classes = useStyles();
  const tooltipCommonClasses = useTooltipCommonStyles();
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses),
    [tooltipCommonClasses]
  );

  const [operatorGraph, setOperatorGraph] = React.useState<OperatorGraph | undefined>(undefined);
  const [operatorTable, setOperatorTable] = React.useState<OperationTableData | undefined>(undefined);
  const [sortColumn, setSortColumn] = React.useState('');
  const [tableTooltips, setTableTooltips] = React.useState<any | undefined>(undefined);
  const [groupBy, setGroupBy] = React.useState(OperationGroupBy.OPERATION);
  const [searchOperatorName, setSearchOperatorName] = React.useState('');
  const [topText, actualTop, useTop, setTopText, setUseTop] = useTopN({
    defaultUseTop: UseTop.USE,
    defaultTop: 10,
  });

  const getName = React.useCallback((row: OperationTableDataInner) => row.name, []);
  const [searchedOperatorTable] = useSearchDirectly(searchOperatorName, getName, operatorTable);

  const onSearchOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOperatorName(event.target.value as string);
  };

  React.useEffect(() => {
    if (operatorGraph) {
      const counts = [
        operatorGraph.device_self_time?.rows.length ?? 0,
        operatorGraph.device_total_time?.rows.length ?? 0,
        operatorGraph.host_self_time.rows?.length ?? 0,
        operatorGraph.host_total_time.rows?.length ?? 0,
      ];
      setTopText(String(Math.min(Math.max(...counts), 10)));
    }
  }, [operatorGraph]);

  React.useEffect(() => {
    api.defaultApi.operationTableGet(run, worker, span, groupBy).then((resp) => {
      setSortColumn(resp.metadata.sort);
      setTableTooltips(resp.metadata.tooltips);
      setOperatorTable(resp.data);
    });
  }, [run, worker, span, groupBy]);

  React.useEffect(() => {
    api.defaultApi.operationGet(run, worker, span, groupBy).then((resp) => {
      setOperatorGraph(resp);
    });
  }, [run, worker, span, groupBy]);

  const onGroupByChanged: SelectProps['onChange'] = (event) => {
    setGroupBy(event.target.value as OperationGroupBy);
  };

  const onUseTopChanged: RadioGroupProps['onChange'] = (event) => {
    setUseTop(event.target.value as UseTop);
  };

  const onTopChanged = (event: React.ChangeEvent<HTMLInputElement>): void => {
    setTopText(event.target.value);
  };

  const inputProps: StandardTextFieldProps['inputProps'] = {
    min: 1,
  };

  const renderCharts = (graph: api.OperatorGraph): JSX.Element => {
    return (
      <GridList className={classes.full} cellHeight='auto' cols={2}>
        {graph.device_self_time && (
          <GridListTile>
            <Card>
              {graph.device_self_time.title && (
                <CardHeader
                  title={chartHeaderRenderer(
                    graph.device_self_time.title,
                    deviceTarget === 'Ascend' ? deviceSelfTimeTooltipAscend : deviceSelfTimeTooltip
                  )}
                />
              )}
              <PieChart graph={graph.device_self_time} top={actualTop} />
            </Card>
          </GridListTile>
        )}
        {graph.device_total_time && (
          <GridListTile>
            <Card>
              {graph.device_total_time.title && (
                <CardHeader
                  title={chartHeaderRenderer(
                    graph.device_total_time.title,
                    deviceTarget === 'Ascend' ? deviceTotalTimeTooltipAscend : deviceTotalTimeTooltip
                  )}
                />
              )}
              <PieChart graph={graph.device_total_time} top={actualTop} />
            </Card>
          </GridListTile>
        )}
        <GridListTile>
          <Card>
            {graph.host_self_time.title && (
              <CardHeader title={chartHeaderRenderer(graph.host_self_time.title, hostSelfTimeTooltip)} />
            )}
            <PieChart graph={graph.host_self_time} top={actualTop} />
          </Card>
        </GridListTile>
        <GridListTile>
          <Card>
            {graph.host_total_time.title && (
              <CardHeader title={chartHeaderRenderer(graph.host_total_time.title, hostTotalTimeTooltip)} />
            )}
            <PieChart graph={graph.host_total_time} top={actualTop} />
          </Card>
        </GridListTile>
      </GridList>
    );
  };

  return (
    <div className={classes.root}>
      <Card variant='outlined'>
        <CardHeader title='Operator View' />
        <CardContent>
          <Grid direction='column' container spacing={1}>
            <Grid container item md={12}>
              <Grid item>
                <RadioGroup row value={useTop} onChange={onUseTopChanged}>
                  <FormControlLabel value={UseTop.NOT_USE} control={<Radio />} label='All operators' />
                  <FormControlLabel value={UseTop.USE} control={<Radio />} label='Top operators to show' />
                </RadioGroup>
              </Grid>
              {useTop === UseTop.USE && (
                <Grid item className={classes.verticalInput}>
                  <TextField
                    classes={{ root: classes.inputWidth }}
                    inputProps={inputProps}
                    type='number'
                    value={topText}
                    onChange={onTopChanged}
                    error={!topIsValid(topText)}
                  />
                </Grid>
              )}
            </Grid>
            <Grid container item md={12}>
              <DataLoading value={operatorGraph}>{renderCharts}</DataLoading>
            </Grid>
            <Grid item container direction='column' spacing={1}>
              <Grid item>
                <Grid container justify='space-around'>
                  <Grid item>
                    <InputLabel id='operator-group-by'>Group By</InputLabel>
                    <Select labelId='operator-group-by' value={groupBy} onChange={onGroupByChanged}>
                      <MenuItem value={OperationGroupBy.OPERATION_AND_INPUT_SHAPE}>Operator + Input Shape</MenuItem>
                      <MenuItem value={OperationGroupBy.OPERATION}>Operator</MenuItem>
                    </Select>
                  </Grid>
                  <Grid item>
                    <TextField
                      classes={{ root: classes.inputWidthOverflow }}
                      value={searchOperatorName}
                      onChange={onSearchOperatorChanged}
                      type='search'
                      label='Search by Name'
                      inputProps={{
                        maxLength: 200,
                      }}
                    />
                  </Grid>
                </Grid>
              </Grid>
              <Grid>
                <DataLoading value={searchedOperatorTable}>
                  {(table): JSX.Element => (
                    <OperationTable
                      data={table}
                      groupBy={groupBy}
                      run={run}
                      span={span}
                      worker={worker}
                      sortColumn={sortColumn}
                      tooltips={tableTooltips}
                      deviceTarget={deviceTarget}
                    />
                  )}
                </DataLoading>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  );
};
