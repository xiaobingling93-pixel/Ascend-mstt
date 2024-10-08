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
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Radio from '@material-ui/core/Radio';
import RadioGroup, { RadioGroupProps } from '@material-ui/core/RadioGroup';
import Select, { SelectProps } from '@material-ui/core/Select';
import { makeStyles } from '@material-ui/core/styles';
import TextField, {
  StandardTextFieldProps,
  TextFieldProps,
} from '@material-ui/core/TextField';
import * as React from 'react';
import * as api from '../api';
import { Graph } from '../api';
import { KernelGroupBy } from '../constants/groupBy';
import { useSearch } from '../utils/search';
import { topIsValid, UseTop, useTopN } from '../utils/top';
import { AntTableChart } from './charts/AntTableChart';
import { PieChart } from './charts/PieChart';
import { DataLoading } from './DataLoading';
import { makeChartHeaderRenderer, useTooltipCommonStyles } from './helpers';
import {
  GPUKernelTotalTimeTooltip,
  TensorCoresPieChartTooltip,
  TensorCoresPieChartTooltipAscend,
} from './TooltipDescriptions';

export interface IProps {
  run: string;
  worker: string;
  span: string;
  deviceTarget: string;
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
}));

export const Kernel: React.FC<IProps> = (props) => {
  const { run, worker, span, deviceTarget } = props;
  const classes = useStyles();
  const tooltipCommonClasses = useTooltipCommonStyles();
  const chartHeaderRenderer = React.useMemo(
    () => makeChartHeaderRenderer(tooltipCommonClasses),
    [tooltipCommonClasses]
  );

  const [kernelGraph, setKernelGraph] = React.useState<Graph | undefined>(
    undefined
  );
  const [tcGraph, setTcGraph] = React.useState<Graph | undefined>(undefined);
  const [kernelTable, setKernelTable] = React.useState<Graph | undefined>(
    undefined
  );
  const [groupBy, setGroupBy] = React.useState(KernelGroupBy.Kernel);
  const [searchKernelName, setSearchKernelName] = React.useState('');
  const [searchOpName, setSearchOpName] = React.useState('');
  const [sortColumn, setSortColumn] = React.useState('');
  const [hasStep, setHasStep] = React.useState(false);

  const [topText, actualTop, useTop, setTopText, setUseTop] = useTopN({
    defaultUseTop: UseTop.Use,
    defaultTop: 10,
  });

  React.useEffect(() => {
    setSearchOpName('');
  }, [groupBy]);

  React.useEffect(() => {
    if (kernelGraph) {
      setTopText(String(Math.min(kernelGraph.rows?.length, 10)));
    }
  }, [kernelGraph]);

  React.useEffect(() => {
    api.defaultApi.kernelTableGet(run, worker, span, groupBy).then((resp) => {
      setSortColumn(resp.metadata.sort);
      setKernelTable(resp.data);
      const nameColumnIdx = resp.data.columns.findIndex(
        (c) => c.name.toLowerCase() === 'step id'
      );
      setHasStep(nameColumnIdx > -1);
    });
  }, [run, worker, span, groupBy]);

  React.useEffect(() => {
    api.defaultApi
      .kernelGet(run, worker, span, KernelGroupBy.Kernel)
      .then((resp) => {
        setKernelGraph(resp.total);
        setGroupBy(
          resp.device_target === 'Ascend'
            ? KernelGroupBy.KernelNameAndOpName
            : KernelGroupBy.Kernel
        );
      });
  }, [run, worker, span]);

  React.useEffect(() => {
    api.defaultApi.kernelTcPieGet(run, worker, span).then((resp) => {
      setTcGraph(resp.total);
    });
  }, [run, worker, span]);

  const [searchedKernelTable] = useSearch(
    searchKernelName,
    'name',
    kernelTable
  );
  const [searchedOpTable] = useSearch(
    searchOpName,
    deviceTarget === 'Ascend' ? 'step id' : 'operator',
    searchedKernelTable
  );

  const onGroupByChanged: SelectProps['onChange'] = (event) => {
    setGroupBy(event.target.value as KernelGroupBy);
  };

  const onSearchKernelChanged: TextFieldProps['onChange'] = (event) => {
    setSearchKernelName(event.target.value as string);
  };

  const onSearchOpChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOpName(event.target.value as string);
  };

  const onUseTopChanged: RadioGroupProps['onChange'] = (event) => {
    setUseTop(event.target.value as UseTop);
  };

  const onTopChanged = (event: React.ChangeEvent<HTMLInputElement>) => {
    setTopText(event.target.value);
  };

  const inputProps: StandardTextFieldProps['inputProps'] = {
    min: 1,
  };

  const GPUKernelTotalTimeTitle = React.useMemo(
    () => chartHeaderRenderer('Total Time (us)', GPUKernelTotalTimeTooltip),
    [chartHeaderRenderer]
  );

  const TensorCoresTitle = React.useMemo(
    () =>
      deviceTarget === 'Ascend'
        ? chartHeaderRenderer(
            'Accelerator Core Utilization',
            TensorCoresPieChartTooltipAscend
          )
        : chartHeaderRenderer(
            'Tensor Cores Utilization',
            TensorCoresPieChartTooltip
          ),
    [chartHeaderRenderer, deviceTarget]
  );

  return (
    <div className={classes.root}>
      <Card variant='outlined'>
        <CardHeader title='Kernel View' />
        <CardContent>
          <Grid container spacing={1}>
            <Grid item container sm={12}>
              <Grid item>
                <RadioGroup row value={useTop} onChange={onUseTopChanged}>
                  <FormControlLabel
                    value={UseTop.NotUse}
                    control={<Radio />}
                    label='All kernels'
                  />
                  <FormControlLabel
                    value={UseTop.Use}
                    control={<Radio />}
                    label='Top kernels to show'
                  />
                </RadioGroup>
              </Grid>
              {useTop === UseTop.Use && (
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
            <Grid item sm={6}>
              <DataLoading value={kernelGraph}>
                {(graph) => (
                  <Card elevation={0}>
                    <CardHeader title={GPUKernelTotalTimeTitle} />
                    <PieChart
                      title={graph.title}
                      graph={graph}
                      top={actualTop}
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>
            <Grid item sm={6}>
              <DataLoading value={tcGraph}>
                {(graph) => (
                  <Card elevation={0}>
                    <CardHeader title={TensorCoresTitle} />
                    <PieChart
                      title={graph.title}
                      graph={graph}
                      colors={['#0099C6', '#DD4477', '#66AA00', '#B82E2E']}
                      top={actualTop}
                      tooltip_mode='percentage'
                    />
                  </Card>
                )}
              </DataLoading>
            </Grid>
            <Grid item container direction='column' spacing={1} sm={12}>
              <Grid item container>
                <Grid sm={6} item container justify='space-around'>
                  <Grid item>
                    <InputLabel id='kernel-group-by'>Group By</InputLabel>
                    <Select
                      labelId='kernel-group-by'
                      value={groupBy}
                      onChange={onGroupByChanged}
                    >
                      <MenuItem value={KernelGroupBy.KernelNameAndOpName}>
                        {deviceTarget === 'Ascend'
                          ? 'Statistic'
                          : 'Kernel Properties + Op Name'}
                      </MenuItem>
                      <MenuItem value={KernelGroupBy.Kernel}>
                        {deviceTarget === 'Ascend' ? 'All' : 'Kernel Name'}
                      </MenuItem>
                    </Select>
                  </Grid>
                </Grid>
                <Grid sm={6} item container spacing={1}>
                  <Grid item>
                    <TextField
                      classes={{ root: classes.inputWidthOverflow }}
                      value={searchKernelName}
                      onChange={onSearchKernelChanged}
                      type='search'
                      label='Search by Name'
                      inputProps={{
                        maxLength: 200,
                      }}
                    />
                  </Grid>
                  {deviceTarget === 'Ascend'
                    ? groupBy === KernelGroupBy.Kernel &&
                      hasStep && (
                        <Grid item>
                          <TextField
                            classes={{ root: classes.inputWidthOverflow }}
                            value={searchOpName}
                            onChange={onSearchOpChanged}
                            type='search'
                            label='Search by Step Id'
                            inputProps={{
                              maxLength: 200,
                            }}
                          />
                        </Grid>
                      )
                    : groupBy === KernelGroupBy.KernelNameAndOpName && (
                        <Grid item>
                          <TextField
                            classes={{ root: classes.inputWidthOverflow }}
                            value={searchOpName}
                            onChange={onSearchOpChanged}
                            type='search'
                            label='Search by Operator Name'
                            inputProps={{
                              maxLength: 200,
                            }}
                          />
                        </Grid>
                      )}
                </Grid>
              </Grid>
              <Grid item container>
                <Grid item sm={12}>
                  <DataLoading value={searchedOpTable}>
                    {(graph) => (
                      <AntTableChart graph={graph} sortColumn={sortColumn} />
                    )}
                  </DataLoading>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </div>
  );
};
