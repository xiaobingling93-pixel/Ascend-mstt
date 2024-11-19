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
import Grid from '@material-ui/core/Grid';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select, { SelectProps } from '@material-ui/core/Select';
import Slider from '@material-ui/core/Slider';
import { makeStyles } from '@material-ui/core/styles';
import TextField, { TextFieldProps } from '@material-ui/core/TextField';
import * as React from 'react';
import * as api from '../api';
import {
  Graph,
  GraphAscend,
  MemoryCurveDataAll,
  MemoryCurveData,
  MemoryCurveDataAscend,
  MemoryEventsData,
  MemoryEventsDataAll,
  MemoryStatsData,
} from '../api';
import { useSearchDirectly } from '../utils/search';
import { AntTableChart } from './charts/AntTableChart';
import { LineChart } from './charts/NewLineChart';
import { DataLoading } from './DataLoading';
import { MemoryStatsTable } from './tables/MemoryStatsTable';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  curve: {
    marginBottom: 20,
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
  filterSlider: {
    marginTop: 15,
    marginRight: 6,
    width: 250,
  },
  filterInput: {
    width: 100,
  },
}));

export interface IProps {
  run: string;
  worker: string;
  span: string;
  deviceTarget: string;
}

const tags = ['Operator', 'Component'];

export const MemoryView: React.FC<IProps> = React.memo((props) => {
  interface EventSizeFilter {
    [deviceName: string]: Array<number>;
  }

  interface MaxEventSize {
    [deviceName: string]: number;
  }

  const { run, worker, span, deviceTarget } = props;
  const classes = useStyles();

  const [memoryStatsData, setMemoryStatsData] = React.useState<
    MemoryStatsData | undefined
  >(undefined);

  // for backward compatability, old profile do not have events to show
  const showEvents = () => {
    return memoryEventsData && Object.keys(memoryEventsData.rows).length !== 0;
  };
  const [memoryEventsData, setMemoryEventsData] = React.useState<
    MemoryEventsData | undefined
  >(undefined);

  // for backward compatability, old profile do not have curve to show
  const showCurve = () => {
    return memoryCurveData && Object.keys(memoryCurveData.rows).length !== 0;
  };
  const [memoryCurveData, setMemoryCurveData] = React.useState<
    MemoryCurveData | MemoryCurveDataAscend | undefined
  >(undefined);

  const [lineChartData, setLineChartData] = React.useState<
    Graph | GraphAscend | undefined
  >(undefined);

  const [devices, setDevices] = React.useState<string[]>([]);
  const [device, setDevice] = React.useState('');
  const [tag, setTag] = React.useState('Operator');
  const memoryCurveDataAllRef = React.useRef<MemoryCurveDataAll | undefined>(
    undefined
  );
  const memoryEventDataAllRef = React.useRef<MemoryEventsDataAll | undefined>(
    undefined
  );

  interface SelectedRange {
    start: number;
    end: number;
    startTs: number;
    endTs: number;
  }
  const [selectedRange, setSelectedRange] = React.useState<
    SelectedRange | undefined
  >();
  const [searchOperatorName, setSearchOperatorName] = React.useState('');
  const [searchEventOperatorName, setSearchEventOperatorName] =
    React.useState('');
  const [filterEventSize, setFilterEventSize] = React.useState<EventSizeFilter>(
    {}
  );
  const [maxSize, setMaxSize] = React.useState<MaxEventSize>({});

  const getSearchIndex = function () {
    if (!memoryStatsData) {
      return -1;
    }
    for (let i = 0; i < memoryStatsData.columns.length; i++) {
      if (memoryStatsData.columns[i].name === memoryStatsData.metadata.search) {
        return i;
      }
    }
    return -1;
  };

  const getStep = (size: number, indexBias: number) => {
    return 10 ** (Math.floor(Math.log10(size !== 0 ? size : 1)) - indexBias);
  };

  const filterByEventSize = <T,>(
    rows: T[] | undefined,
    size: Array<number>
  ) => {
    const result = React.useMemo(() => {
      if (!rows) {
        return undefined;
      }

      // workaround type system
      const field = (row: any): number => {
        const sizeColIndex = 1;
        return row[sizeColIndex];
      };

      return rows.filter((row) => {
        return field(row) >= size[0] && field(row) <= size[1];
      });
    }, [rows, size]);

    return result;
  };

  const searchIndex = getSearchIndex();
  const getName = React.useCallback(
    (row: any) => row[searchIndex],
    [searchIndex]
  );
  const getNameAscend = (row: any) => row[0];
  const [searchedTableDataRows] = useSearchDirectly(
    searchOperatorName,
    getName,
    memoryStatsData?.rows[device] ?? []
  );
  const [searchedEventsTableDataRows] = useSearchDirectly(
    searchEventOperatorName,
    deviceTarget === 'Ascend' ? getNameAscend : getName,
    filterByEventSize(
      memoryEventsData?.rows[device],
      filterEventSize[device] ?? [0, Infinity]
    ) ?? []
  );

  const onSearchOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchOperatorName(event.target.value as string);
  };

  const onSearchEventOperatorChanged: TextFieldProps['onChange'] = (event) => {
    setSearchEventOperatorName(event.target.value as string);
  };

  const [selectedRecord, setSelectedRecord] = React.useState<any | undefined>();
  const onRowSelected = (record?: object, rowIndex?: number) => {
    setSelectedRecord(record);
  };

  const onFilterEventSizeChanged = (
    event: any,
    newValue: number | number[]
  ) => {
    setFilterEventSize({
      ...filterEventSize,
      [device]: newValue as number[],
    });
  };

  const onFilterEventMinSizeInputChanged = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFilterEventSize({
      ...filterEventSize,
      [device]: [Number(event.target.value), filterEventSize[device][1]],
    });
  };

  const onFilterEventMaxSizeInputChanged = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFilterEventSize({
      ...filterEventSize,
      [device]: [filterEventSize[device][0], Number(event.target.value)],
    });
  };

  React.useEffect(() => {
    deviceTarget !== 'Ascend' &&
      api.defaultApi
        .memoryGet(
          run,
          worker,
          span,
          selectedRange?.startTs,
          selectedRange?.endTs
        )
        .then((resp) => {
          setMemoryStatsData(resp);
          if (!devices || devices.length === 0) {
            // setDevices only execute on view load. Since selection on curve
            // might filter all events later, some devices might is missing.
            setDevices(Object.keys(resp.rows));
            setDevice(resp.metadata.default_device);
          }
        });
  }, [run, worker, span, selectedRange]);

  React.useEffect(() => {
    api.defaultApi
      .memoryEventsGet(
        run,
        worker,
        span,
        selectedRange?.startTs,
        selectedRange?.endTs
      )
      .then((resp) => {
        const tempRes =
          deviceTarget === 'Ascend'
            ? (resp as MemoryEventsDataAll).operator
            : (resp as MemoryEventsData);
        if (deviceTarget === 'Ascend') {
          memoryEventDataAllRef.current = resp as MemoryEventsDataAll;
        }
        let curMaxSize: MaxEventSize = {};
        let curFilterEventSize: EventSizeFilter = {};
        Object.keys(tempRes.rows).forEach((deviceName) => {
          curMaxSize[deviceName] = 0;
          for (let i = 0; i < tempRes.rows[deviceName].length; i++) {
            curMaxSize[deviceName] = Math.max(
              curMaxSize[deviceName],
              tempRes.rows[deviceName][i][1]
            );
          }
          curFilterEventSize[deviceName] = [
            curMaxSize[deviceName] / 4,
            curMaxSize[deviceName],
          ];
          curMaxSize[deviceName] = curMaxSize[deviceName];
        });
        setMaxSize(curMaxSize);
        setFilterEventSize(curFilterEventSize);
        setMemoryEventsData(tempRes);
      });
  }, [run, worker, span, selectedRange]);

  React.useEffect(() => {
    api.defaultApi.memoryCurveGet(run, worker, span).then((resp) => {
      // Reset the select range to null whenever run/worker/span changes
      setSelectedRange(undefined);
      if (deviceTarget === 'Ascend') {
        const allCurveData = resp as MemoryCurveDataAll;
        memoryCurveDataAllRef.current = allCurveData;
        setDevice(allCurveData.default_device);
        setDevices(allCurveData.devices);
        setMemoryCurveData(allCurveData.total);
        setTag('Operator');
      } else {
        setMemoryCurveData(resp as MemoryCurveData);
      }
    });
  }, [run, worker, span]);

  React.useEffect(() => {
    if (memoryCurveData !== undefined) {
      if (deviceTarget === 'Ascend') {
        setLineChartData({
          title: memoryCurveData.metadata.peaks[device] ?? '',
          columns: memoryCurveData.columns[device] ?? [],
          rows: memoryCurveData.rows[device] ?? {},
        });
      } else {
        setLineChartData({
          title: memoryCurveData.metadata.peaks[device],
          columns: memoryCurveData.columns,
          rows: memoryCurveData.rows[device] ?? [],
        });
      }
    }
  }, [memoryCurveData, device]);

  const onDeviceChanged: SelectProps['onChange'] = (event) => {
    setDevice(event.target.value as string);
    setSelectedRange(undefined);
  };

  const onTagChanged: SelectProps['onChange'] = (event) => {
    setTag(event.target.value as string);
    if (event.target.value === 'Operator') {
      setMemoryCurveData(memoryCurveDataAllRef.current?.total);
      setMemoryEventsData(memoryEventDataAllRef.current?.operator);
      setSelectedRange(undefined);
    } else {
      setMemoryCurveData(memoryCurveDataAllRef.current?.ptaGe);
      setMemoryEventsData(memoryEventDataAllRef.current?.component);
    }
  };

  const onSelectedRangeChanged = (start: number, end: number) => {
    if (start > end) {
      setSelectedRange(undefined);
      return;
    }

    let allDatas =
      deviceTarget === 'Ascend'
        ? memoryCurveData?.rows[device]?.Allocated
        : memoryCurveData?.rows[device];
    if (allDatas.length <= 1) {
      setSelectedRange(undefined);
      return;
    }

    let startTs = 0;
    let endTs = 0;
    let realStart = 0;
    let realEnd = 0;
    let startId = 1;
    let endId = 0;
    let needLoopStart = true;
    for (let i = 1; i < allDatas.length; i++) {
      if (startId > start && needLoopStart) {
        needLoopStart = false;
        realStart = i - 1;
      }
      if (allDatas[i][0] !== allDatas[i - 1][0]) {
        if (startId <= start) {
          startId += 1;
        }
        endId += 1;
      }
      if (endId > end) {
        realEnd = i - 1;
        break;
      } else {
        realEnd = i;
        if (needLoopStart) {
          realStart = i;
        }
      }
    }

    if (deviceTarget === 'Ascend') {
      startTs = allDatas[realStart][0];
      endTs = allDatas[realEnd][0];
    } else {
      let bias = memoryCurveData?.metadata.first_ts ?? 0;
      let scale = 1 / (memoryCurveData?.metadata.time_factor ?? 1);
      startTs = Math.round(allDatas[realStart][0] * scale + bias);
      endTs = Math.round(allDatas[realEnd][0] * scale + bias);
    }

    setSelectedRange({ start, end, startTs, endTs });
  };

  return (
    <div className={classes.root}>
      <Card variant='outlined'>
        <CardHeader title='Memory View' />
        <CardContent>
          <Grid direction='column' container spacing={1}>
            <Grid item className={classes.curve}>
              <DataLoading value={memoryCurveData}>
                {(graph) => (
                  <Grid container direction='column'>
                    <Grid container>
                      <Grid item sm={3}>
                        <InputLabel id='memory-curve-device'>Device</InputLabel>
                        <Select
                          labelId='memory-curve-device'
                          value={device}
                          onChange={onDeviceChanged}
                        >
                          {devices.map((item) => (
                            <MenuItem value={item}>{item}</MenuItem>
                          ))}
                        </Select>
                      </Grid>
                      {deviceTarget === 'Ascend' && (
                        <Grid item>
                          <InputLabel id='memory-curve-tag'>
                            Group By
                          </InputLabel>
                          <Select
                            labelId='memory-curve-tag'
                            value={tag}
                            onChange={onTagChanged}
                          >
                            {tags.map((item) => (
                              <MenuItem value={item}>{item}</MenuItem>
                            ))}
                          </Select>
                        </Grid>
                      )}
                    </Grid>
                    {showCurve() &&
                      lineChartData &&
                      lineChartData.columns.length > 0 && (
                        <Grid item>
                          <div>
                            <LineChart
                              hAxisTitle={`Time (${graph.metadata.time_metric})`}
                              vAxisTitle={`Memory Usage (${graph.metadata.memory_metric})`}
                              graph={lineChartData}
                              deviceTarget={deviceTarget}
                              tag={tag}
                              onSelectionChanged={
                                tag !== 'Component'
                                  ? onSelectedRangeChanged
                                  : undefined
                              }
                              record={selectedRecord}
                            />
                          </div>
                        </Grid>
                      )}
                  </Grid>
                )}
              </DataLoading>
            </Grid>
            {showEvents() && (
              <>
                {(deviceTarget !== 'Ascend' || tag === 'Operator') && (
                  <Grid container>
                    <Grid item container sm={6} justifyContent='space-around'>
                      <Grid item>
                        <TextField
                          classes={{ root: classes.inputWidthOverflow }}
                          value={searchEventOperatorName}
                          onChange={onSearchEventOperatorChanged}
                          type='search'
                          label='Search by Name'
                          inputProps={{
                            maxLength: 200,
                          }}
                        />
                      </Grid>
                    </Grid>
                    <Grid item sm={6}>
                      <Grid container direction='row' spacing={2}>
                        <Grid item>
                          <TextField
                            className={classes.filterInput}
                            label='Min Size(KB)'
                            value={filterEventSize[device]?.[0] ?? 0}
                            onChange={onFilterEventMinSizeInputChanged}
                            inputProps={{
                              step: getStep(maxSize[device] ?? 0, 3),
                              min: 0,
                              max: filterEventSize[device]?.[1] ?? 0,
                              type: 'number',
                              'aria-labelledby': 'input-slider',
                            }}
                          />
                        </Grid>
                        <Grid item>
                          <Slider
                            className={classes.filterSlider}
                            value={filterEventSize[device] ?? [0, 0]}
                            onChange={onFilterEventSizeChanged}
                            aria-labelledby='input-slider'
                            min={0}
                            max={maxSize[device] ?? 0}
                            step={getStep(maxSize[device] ?? 0, 5)}
                          />
                        </Grid>
                        <Grid item>
                          <TextField
                            className={classes.filterInput}
                            label='Max Size(KB)'
                            value={filterEventSize[device]?.[1] ?? 0}
                            onChange={onFilterEventMaxSizeInputChanged}
                            inputProps={{
                              step: getStep(maxSize[device] ?? 0, 3),
                              min: filterEventSize[device]?.[0] ?? 0,
                              max: maxSize[device] ?? 0,
                              type: 'number',
                              'aria-labelledby': 'input-slider',
                            }}
                          />
                        </Grid>
                      </Grid>
                    </Grid>
                  </Grid>
                )}
                <Grid item direction='column'>
                  <DataLoading value={memoryEventsData}>
                    {(data) => {
                      return (
                        <AntTableChart
                          graph={{
                            columns: data.columns,
                            rows:
                              tag === 'Component'
                                ? data.rows[device] ?? []
                                : searchedEventsTableDataRows ?? [],
                          }}
                          initialPageSize={10}
                          onRowSelected={onRowSelected}
                        />
                      );
                    }}
                  </DataLoading>
                </Grid>
              </>
            )}
            {deviceTarget !== 'Ascend' && (
              <>
                <Grid item container direction='column' sm={6}>
                  <Grid item container direction='column' alignContent='center'>
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
                <Grid item direction='column'>
                  <DataLoading value={memoryStatsData}>
                    {(data) => (
                      <MemoryStatsTable
                        data={{
                          rows: searchedTableDataRows,
                          columns: data.columns,
                        }}
                        sort={memoryStatsData!.metadata.sort}
                      />
                    )}
                  </DataLoading>
                </Grid>
              </>
            )}
          </Grid>
        </CardContent>
      </Card>
    </div>
  );
});
