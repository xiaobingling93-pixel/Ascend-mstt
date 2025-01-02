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

import Box from '@material-ui/core/Box';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import CardHeader from '@material-ui/core/CardHeader';
import ClickAwayListener from '@material-ui/core/ClickAwayListener';
import CssBaseline from '@material-ui/core/CssBaseline';
import Divider from '@material-ui/core/Divider';
import Drawer from '@material-ui/core/Drawer';
import Fab from '@material-ui/core/Fab';
import FormControl from '@material-ui/core/FormControl';
import IconButton from '@material-ui/core/IconButton';
import ListSubheader from '@material-ui/core/ListSubheader';
import MenuItem from '@material-ui/core/MenuItem';
import Select, { SelectProps } from '@material-ui/core/Select';
import { makeStyles } from '@material-ui/core/styles';
import Tab from '@material-ui/core/Tab';
import Tabs from '@material-ui/core/Tabs';
import Typography from '@material-ui/core/Typography';
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft';
import ChevronRightIcon from '@material-ui/icons/ChevronRight';
import { message } from 'antd';
import 'antd/es/button/style/css';
import 'antd/es/list/style/css';
import 'antd/es/table/style/css';
import clsx from 'clsx';
import * as React from 'react';
import * as api from './api';
import { AccuracyLeftPanel } from './components/Accuracy/AccuracyLeftPanel';
import { FileInfo } from './components/Accuracy/entity';
import { LossComparison } from './components/Accuracy/LossComparison';
import { DiffOverview } from './components/DiffOverview';
import { DistributedView } from './components/DistributedView';
import { FullCircularProgress } from './components/FullCircularProgress';
import { Kernel as KernelView } from './components/Kernel';
import { MemoryView } from './components/MemoryView';
import { ModuleView } from './components/ModuleView';
import { Operator as OperatorView } from './components/Operator';
import { Overview as OverviewPage } from './components/Overview';
import { TraceView } from './components/TraceView';
import { setup } from './setup';
import './styles.css';
import { firstOrUndefined, sleep } from './utils';

export enum Views {
  Overview = 'Overview',
  Operator = 'Operator',
  Kernel = 'Kernel',
  Trace = 'Trace',
  Distributed = 'Distributed',
  Memory = 'Memory',
  Module = 'Module',
  Lightning = 'Lightning',
}

const viewNames = {
  [Views.Overview]: Views.Overview,
  [Views.Operator]: Views.Operator,
  [Views.Kernel]: 'Kernel',
  [Views.Trace]: Views.Trace,
  [Views.Distributed]: Views.Distributed,
  [Views.Memory]: Views.Memory,
  [Views.Module]: Views.Module,
  [Views.Lightning]: Views.Lightning,
};

const drawerWidth = 340;
const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
    height: '100%',
  },
  appBar: {
    zIndex: theme.zIndex.drawer + 1,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
  },
  appBarShift: {
    marginLeft: drawerWidth,
    width: `calc(100% - ${drawerWidth}px)`,
    transition: theme.transitions.create(['width', 'margin'], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
  },
  menuButton: {
    marginRight: 36,
  },
  hide: {
    display: 'none',
  },
  drawer: {
    width: drawerWidth,
    flexShrink: 0,
    whiteSpace: 'nowrap',
  },
  drawerOpen: {
    width: drawerWidth,
    zIndex: 999,
    transition: theme.transitions.create('width', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
  },
  drawerClose: {
    transition: theme.transitions.create('width', {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.leavingScreen,
    }),
    overflowX: 'hidden',
    width: 0,
    [theme.breakpoints.up('sm')]: {
      width: 0,
    },
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    padding: theme.spacing(0, 1),
    // necessary for content to be below app bar
    ...theme.mixins.toolbar,
  },
  content: {
    flexGrow: 1,
    padding: theme.spacing(3),
    overflowX: 'hidden',
  },
  formControl: {
    margin: theme.spacing(1),
    minWidth: 120,
  },
  fab: {
    marginLeft: theme.spacing(1),
    marginTop: theme.spacing(1),
    position: 'absolute',
  },
  iconButton: {
    padding: '8px',
  },
}));

export const App = (): JSX.Element => {
  const classes = useStyles();

  // #region - State
  const [selectedTab, setSelectedTab] = React.useState(0);

  const [run, setRun] = React.useState<string>('');
  const [runs, setRuns] = React.useState<string[]>([]);
  const [runsLoading, setRunsLoading] = React.useState(true);

  const [workers, setWorkers] = React.useState<string[]>([]);
  const [worker, setWorker] = React.useState<string>('');

  const [spans, setSpans] = React.useState<string[]>([]);
  const [span, setSpan] = React.useState<string | ''>('');

  const [views, setViews] = React.useState<Views[]>([]);
  const [view, setView] = React.useState<Views | ''>('');
  const [loaded, setLoaded] = React.useState(false);
  const iframeRef = React.useRef<HTMLIFrameElement>(null);
  const [deviceTarget, setDeviceTarget] = React.useState<string>('GPU');

  const [diffLeftWorkerOptions, setDiffLeftWorkerOptions] = React.useState<string[]>([]);
  const [diffLeftSpansOptions, setDiffLeftSpansOptions] = React.useState<string[]>([]);
  const [diffLeftRun, setDiffLeftRun] = React.useState<string>('');
  const [diffLeftWorker, setDiffLeftWorker] = React.useState<string>('');
  const [diffLeftSpan, setDiffLeftSpan] = React.useState<string | ''>('');

  const [diffRightWorkerOptions, setDiffRightWorkerOptions] = React.useState<string[]>([]);
  const [diffRightSpansOptions, setDiffRightSpansOptions] = React.useState<string[]>([]);
  const [diffRightRun, setDiffRightRun] = React.useState<string>('');
  const [diffRightWorker, setDiffRightWorker] = React.useState<string>('');
  const [diffRightSpan, setDiffRightSpan] = React.useState<string | ''>('');

  const [open, setOpen] = React.useState(true);

  const [topTab, setTopTab] = React.useState<number>(0);
  const [fileList, setFileList] = React.useState<FileInfo[]>([]);
  const [uploadedCount, setUploadedCount] = React.useState<number>(0); // #endregion

  React.useEffect(() => {
    setup()
      .catch(() => {
        message.warning('google chart is not supported offline');
      })
      .finally(() => {
        setLoaded(true);
      });
  }, []);

  const continuouslyFetchRuns = async (): Promise<never> => {
    while (true) {
      try {
        const result = await api.defaultApi.runsGet();
        setRuns(result.runs);
        setRunsLoading(result.loading);
      } catch (e) {
        message.warning(`Cannot fetch runs: ${e}`);
      }
      await sleep(5000);
    }
  };

  React.useEffect(() => {
    continuouslyFetchRuns();
  }, []);

  React.useEffect(() => {
    if (!run || !runs.includes(run)) {
      setRun(firstOrUndefined(runs) ?? '');
    }
  }, [runs]); // #region - Diff Left

  React.useEffect(() => {
    if (diffLeftRun) {
      api.defaultApi.workersGet(diffLeftRun, Views.Overview).then((data) => {
        setDiffLeftWorkerOptions(data);
      });
    }
  }, [diffLeftRun]);

  React.useEffect(() => {
    if (diffLeftRun && diffLeftWorker) {
      api.defaultApi.spansGet(diffLeftRun, diffLeftWorker).then((data) => {
        setDiffLeftSpansOptions(data);
      });
    }
  }, [diffLeftRun, diffLeftWorker]);

  // #endregion
  // #region - Diff Right
  React.useEffect(() => {
    if (diffRightRun) {
      api.defaultApi.workersGet(diffRightRun, Views.Overview).then((data) => {
        setDiffRightWorkerOptions(data);
      });
    }
  }, [diffRightRun]);

  React.useEffect(() => {
    if (diffRightRun && diffRightWorker) {
      api.defaultApi.spansGet(diffRightRun, diffRightWorker).then((data) => {
        setDiffRightSpansOptions(data);
      });
    }
  }, [diffRightRun, diffRightWorker]);

  // #endregion
  // #region - normal
  React.useEffect(() => {
    if (run) {
      api.defaultApi.viewsGet(run).then((rawViews) => {
        const result = rawViews.views.map((v) => Views[Views[v as Views]]).filter(Boolean);
        setDeviceTarget(rawViews.device_target);
        setViews(result);
      });
    }
  }, [run]);

  React.useEffect(() => {
    setView(firstOrUndefined(views) ?? '');
  }, [views]);

  React.useEffect(() => {
    if (run && view) {
      api.defaultApi.workersGet(run, view).then((data) => {
        setWorkers(data);
      });
    }
  }, [run, view]);

  React.useEffect(() => {
    setWorker(firstOrUndefined(workers) ?? '');
  }, [workers]);

  React.useEffect(() => {
    if (run && worker) {
      api.defaultApi.spansGet(run, worker).then((data) => {
        setSpans(data);
      });
    }
  }, [run, worker]);

  React.useEffect(() => {
    setSpan(firstOrUndefined(spans) ?? '');
  }, [spans]);

  // #endregion

  // #region - Event Handler
  const handleTabChange = (event: React.ChangeEvent<Record<string, unknown>>, value: any): void => {
    setSelectedTab(value as number);
  };

  const handleTopTabChange = (event: React.ChangeEvent<Record<string, unknown>>, value: any): void => {
    setTopTab(value as number);
  };

  const handleRunChange: SelectProps['onChange'] = (event) => {
    setRun(event.target.value as string);
    setView('');
    setWorker('');
    setSpan('');
  };

  const handleViewChange: SelectProps['onChange'] = (event) => {
    setView(event.target.value as Views);
    setWorker('');
    setSpan('');
  };

  const handleWorkerChange: SelectProps['onChange'] = (event) => {
    setWorker(event.target.value as string);
    setSpan('');
  };

  const handleSpanChange: SelectProps['onChange'] = (event) => {
    setSpan(event.target.value as string);
  };

  const handleDiffLeftRunChange: SelectProps['onChange'] = (event) => {
    setDiffLeftRun(event.target.value as string);
    setDiffLeftWorker('');
    setDiffLeftSpan('');
  };

  const handleDiffLeftWorkerChange: SelectProps['onChange'] = (event) => {
    setDiffLeftWorker(event.target.value as string);
    setDiffLeftSpan('');
  };

  const handleDiffLeftSpanChange: SelectProps['onChange'] = (event) => {
    setDiffLeftSpan(event.target.value as string);
  };

  const handleDiffRightRunChange: SelectProps['onChange'] = (event) => {
    setDiffRightRun(event.target.value as string);
    setDiffRightWorker('');
    setDiffRightSpan('');
  };

  const handleDiffRightWorkerChange: SelectProps['onChange'] = (event) => {
    setDiffRightWorker(event.target.value as string);
    setDiffRightSpan('');
  };

  const handleDiffRightSpanChange: SelectProps['onChange'] = (event) => {
    setDiffRightSpan(event.target.value as string);
  };

  const handleDrawerOpen = (): void => {
    setOpen(true);
    setIframeActive();
  };

  const handleDrawerClose = (): void => {
    setOpen(false);
    setIframeActive();
  };

  const setIframeActive = (): void => {
    iframeRef.current?.focus();
  };

  const _changeFileList = (files: FileInfo[]): void => {
    if (JSON.stringify(files) !== JSON.stringify(fileList)) {
      setFileList(files);
    }
  };

  const _getViews = (viewName: Views): string => {
    if (viewName === Views.Kernel) {
      return deviceTarget === 'Ascend' ? `NPU ${viewNames[viewName]}` : `GPU ${viewNames[viewName]}`;
    } else {
      return viewNames[viewName];
    }
  };

  const _changeUploadCount = (count: number): void => {
    setUploadedCount(count);
  }; // #endregion

  const renderContent = (): JSX.Element => {
    if (!runsLoading && runs.length === 0) {
      return (
        <Card variant='outlined'>
          <CardHeader title='No Runs Found'></CardHeader>
          <CardContent>
            <Typography>There are not any runs in the log folder.</Typography>
          </CardContent>
        </Card>
      );
    }
    const notReady = !loaded || !run || !worker || !view || !span;
    if (notReady) {
      return <FullCircularProgress />;
    }

    if (selectedTab === 0) {
      switch (view) {
        case Views.Overview:
          return <OverviewPage run={run} worker={worker} span={span} />;
        case Views.Operator:
          return <OperatorView run={run} worker={worker} span={span} deviceTarget={deviceTarget} />;
        case Views.Kernel:
          return <KernelView run={run} worker={worker} span={span} deviceTarget={deviceTarget} />;
        case Views.Trace:
          return <TraceView run={run} worker={worker} span={span} iframeRef={iframeRef} />;
        case Views.Distributed:
          return <DistributedView run={run} worker={worker} span={span} />;
        case Views.Memory:
          return <MemoryView run={run} worker={worker} span={span} deviceTarget={deviceTarget} />;
        case Views.Module:
        case Views.Lightning:
          return <ModuleView run={run} worker={worker} span={span} />;
        default:
          return <></>;
      }
    } else {
      return (
        <DiffOverview
          run={diffLeftRun}
          worker={diffLeftWorker}
          span={diffLeftSpan}
          expRun={diffRightRun}
          expWorker={diffRightWorker}
          expSpan={diffRightSpan}
        />
      );
    }
  };

  const spanComponent = (): JSX.Element => {
    const spanFragment = (
      <React.Fragment>
        <ListSubheader>Spans</ListSubheader>
        <ClickAwayListener onClickAway={setIframeActive}>
          <FormControl variant='outlined' className={classes.formControl}>
            <Select value={span} onChange={handleSpanChange}>
              {spans.map((item) => (
                <MenuItem value={item}>{item}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </ClickAwayListener>
      </React.Fragment>
    );

    if (!spans || spans.length <= 1) {
      return <div className={classes.hide}>{spanFragment}</div>;
    } else {
      return spanFragment;
    }
  };

  return (
    <div className={classes.root}>
      <CssBaseline />
      <Drawer
        variant='permanent'
        anchor='left'
        className={clsx(classes.drawer, {
          [classes.drawerOpen]: open,
          [classes.drawerClose]: !open,
        })}
        classes={{
          paper: clsx({
            [classes.drawerOpen]: open,
            [classes.drawerClose]: !open,
          }),
        }}
        onClick={setIframeActive}
      >
        <div className={classes.toolbar}>
          <IconButton className={classes.iconButton} onClick={handleDrawerClose}>
            <ChevronLeftIcon />
          </IconButton>
        </div>
        <Divider />
        <Box>
          <Tabs value={topTab} onChange={handleTopTabChange} aria-label='top tabs example'>
            <Tab label='Profiling' />
            <Tab label='Accuracy' />
          </Tabs>
        </Box>
        {topTab === 0 ? (
          <>
            <Box>
              <Tabs value={selectedTab} onChange={handleTabChange} aria-label='basic tabs example'>
                <Tab label='Normal' />
                <Tab label='Diff' />
              </Tabs>
            </Box>
            {selectedTab === 0 ? (
              <>
                <ListSubheader>Runs</ListSubheader>
                <ClickAwayListener onClickAway={setIframeActive}>
                  <FormControl variant='outlined' className={classes.formControl}>
                    <Select value={run} onChange={handleRunChange}>
                      {runs.map((item) => (
                        <MenuItem value={item}>{item}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </ClickAwayListener>
                <ListSubheader>Views</ListSubheader>
                <ClickAwayListener onClickAway={setIframeActive}>
                  <FormControl variant='outlined' className={classes.formControl}>
                    <Select value={view} onChange={handleViewChange}>
                      {views.map((item) => (
                        <MenuItem value={item}>{_getViews(item)}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </ClickAwayListener>
                <ListSubheader>Workers</ListSubheader>
                <ClickAwayListener onClickAway={setIframeActive}>
                  <FormControl variant='outlined' className={classes.formControl}>
                    <Select value={worker} onChange={handleWorkerChange}>
                      {workers.map((worker1) => (
                        <MenuItem value={worker1}>{worker1}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </ClickAwayListener>
                {spanComponent()}
              </>
            ) : (
              <>
                <Typography variant='h6'>&nbsp;&nbsp;Baseline</Typography>
                <ListSubheader>Runs</ListSubheader>
                <FormControl variant='outlined' className={classes.formControl}>
                  <Select value={diffLeftRun} onChange={handleDiffLeftRunChange}>
                    {runs.map((item) => (
                      <MenuItem value={item}>{item}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <ListSubheader>Workers</ListSubheader>

                <FormControl variant='outlined' className={classes.formControl}>
                  <Select value={diffLeftWorker} onChange={handleDiffLeftWorkerChange}>
                    {diffLeftWorkerOptions.map((worker2) => (
                      <MenuItem value={worker2}>{worker2}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <ListSubheader>Spans</ListSubheader>
                <FormControl variant='outlined' className={classes.formControl}>
                  <Select value={diffLeftSpan} onChange={handleDiffLeftSpanChange}>
                    {diffLeftSpansOptions.map((span1) => (
                      <MenuItem value={span1}>{span1}</MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Divider />

                <Typography variant='h6'>&nbsp;&nbsp;Experimental</Typography>
                <ListSubheader>Runs</ListSubheader>
                <FormControl variant='outlined' className={classes.formControl}>
                  <Select value={diffRightRun} onChange={handleDiffRightRunChange}>
                    {runs.map((item) => (
                      <MenuItem value={item}>{item}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <ListSubheader>Workers</ListSubheader>
                <FormControl variant='outlined' className={classes.formControl}>
                  <Select value={diffRightWorker} onChange={handleDiffRightWorkerChange}>
                    {diffRightWorkerOptions.map((worker3) => (
                      <MenuItem value={worker3}>{worker3}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <ListSubheader>Spans</ListSubheader>
                <FormControl variant='outlined' className={classes.formControl}>
                  <Select value={diffRightSpan} onChange={handleDiffRightSpanChange}>
                    {diffRightSpansOptions.map((span2) => (
                      <MenuItem value={span2}>{span2}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </>
            )}
          </>
        ) : (
          <AccuracyLeftPanel onChangeCheckedFileList={_changeFileList} onChangeUploadedCount={_changeUploadCount} />
        )}
      </Drawer>
      {!open && (
        <Fab className={classes.fab} size='small' color='primary' aria-label='show menu' onClick={handleDrawerOpen}>
          <ChevronRightIcon />
        </Fab>
      )}
      <main className={classes.content}>
        {topTab === 0 ? renderContent() : <LossComparison fileList={fileList} fileCount={uploadedCount} />}
      </main>
    </div>
  );
};
