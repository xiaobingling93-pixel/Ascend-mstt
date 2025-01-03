/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/
import Card from '@material-ui/core/Card';
import CardHeader from '@material-ui/core/CardHeader';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Select, { SelectProps } from '@material-ui/core/Select';
import { makeStyles } from '@material-ui/core/styles';
import { message, Table } from 'antd';
import * as React from 'react';
import { FlameGraph } from 'react-flame-graph';
import { defaultApi, KeyedColumn, ModuleStats, ModuleViewData, OperatorNode } from '../api';

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1,
  },
  hide: {
    display: 'none',
  },
}));

export interface IProps {
  run: string;
  worker: string;
  span: string;
}

const getKeyedTableColumns = (columns: KeyedColumn[]): any[] => {
  return columns.map((col) => {
    return {
      dataIndex: col.key,
      key: col.key,
      title: col.name,
    };
  });
};

const getTableRows = (key: number, rows: ModuleStats[]): any[] => {
  let initialKey = key;
  return rows.map((row) => {
    const currentKey = initialKey++;
    const data: any = {
      key: currentKey,
      name: row.name,
      occurences: row.occurences,
      operators: row.operators,
      host_duration: row.host_duration,
      self_host_duration: row.self_host_duration,
      device_duration: row.device_duration,
      self_device_duration: row.self_device_duration,
    };

    if (row.children.length) {
      data.children = getTableRows(key, row.children);
    }

    return data;
  });
};

const getFlameGraphData = (rows: ModuleStats[]): any[] => {
  return rows.map((row) => {
    const data: any = {
      name: row.name,
      value: row.avg_duration,
      tooltip: `${row.name} (module id: ${row.id}): ${row.avg_duration} us`,
    };

    if (row.children.length) {
      data.children = getFlameGraphData(row.children);
    }

    return data;
  });
};

const getTreeHeight = (row: ModuleStats): number => {
  if (row.children?.length) {
    return 1 + Math.max(...row.children.map((child) => getTreeHeight(child)));
  } else {
    return 1;
  }
};

const getOperatorTree = (level: number, row: OperatorNode, result: object[]): void => {
  result.push({
    level: level,
    name: row.name,
    start: row.start_time,
    end: row.end_time,
  });
  if (row.children.length) {
    row.children.forEach((child) => getOperatorTree(level + 1, child, result));
  }
};

export const ModuleView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props;
  const classes = useStyles();

  const [moduleView, setModuleView] = React.useState<ModuleViewData | undefined>(undefined);
  const [flameData, setFlameData] = React.useState<any[]>([]);
  const [flameHeight, setFlameHeight] = React.useState<number>(0);
  const [modules, setModules] = React.useState<number[]>([]);
  const [module, setModule] = React.useState<number>(0);

  const [columns, setColumns] = React.useState<any[]>([]);
  const [rows, setRows] = React.useState<any[]>([]);

  const cardRef = React.useRef<HTMLDivElement>(null);
  const [cardWidth, setCardWidth] = React.useState<number | undefined>(undefined);
  const timelineRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    defaultApi
      .moduleGet(run, worker, span)
      .then((resp) => {
        setModuleView(resp);
        if (resp) {
          // set the flamegraph data
          const flameGraphData: any[] = getFlameGraphData(resp.data);
          setFlameData(flameGraphData);
          const flameGraphHeight = Math.max(...flameGraphData.map((x) => getTreeHeight(x)));
          setFlameHeight(flameGraphHeight * 25);
          setModules(Array.from(Array(flameGraphData.length).keys()));
          setModule(0);

          // set the tree table data
          setColumns(getKeyedTableColumns(resp.columns));
          setRows(getTableRows(1, resp.data));
        }
      })
      .catch((e) => {
        if (e.status === 404) {
          setModules([]);
          setFlameData([]);
          setRows([]);
        }
      });

    if (cardRef.current) {
      setCardWidth(cardRef.current.offsetWidth - 10);
    }
    try {
      if (timelineRef.current) {
        defaultApi.treeGet(run, worker, span).then((resp) => {
          if (resp) {
            const data = new google.visualization.DataTable();
            data.addColumn({ type: 'string', id: 'Layer' });
            data.addColumn({ type: 'string', id: 'Name' });
            data.addColumn({ type: 'string', role: 'tooltip' });
            data.addColumn({ type: 'number', id: 'Start' });
            data.addColumn({ type: 'number', id: 'End' });

            let timelineData: any[] = [];
            getOperatorTree(0, resp, timelineData);
            timelineData.sort((a, b) => a.level - b.level);
            const maxLevel = timelineData[timelineData.length - 1].level;
            timelineData.forEach((d) => {
              data.addRow([
                d.level.toString(),
                d.name,
                `${d.name} Duration: ${d.end - d.start} us`,
                d.start / 1000.0, // the time unit is us returned from server, but the google charts only accept milliseconds here
                d.end / 1000.0,
              ]);
            });

            const chart = new google.visualization.Timeline(timelineRef.current);
            const options = {
              height: (maxLevel + 1) * 50,
              tooltip: {
                isHtml: true,
              },
              timeline: {
                showRowLabels: false,
              },
            };
            chart.draw(data, options);
          }
        });
      }
    } catch (e) {
      message.warning('Timeline in module view is not supported offline.');
    }
  }, [run, worker, span]);

  const handleModuleChange: SelectProps['onChange'] = (event) => {
    setModule(event.target.value as number);
  };

  const moduleComponent = (): JSX.Element => {
    const moduleFragment = (
      <React.Fragment>
        <InputLabel id='module-graph'>Module</InputLabel>
        <Select value={module} onChange={handleModuleChange}>
          {modules.map((m) => (
            <MenuItem value={m}>{m}</MenuItem>
          ))}
        </Select>
      </React.Fragment>
    );

    if (!modules || modules.length <= 1) {
      return <div className={classes.hide}>{moduleFragment}</div>;
    } else {
      return moduleFragment;
    }
  };

  return (
    <div className={classes.root}>
      <Card variant='outlined' ref={cardRef}>
        <CardHeader title='Module View' />
        {rows && rows.length > 0 && (
          <Table
            size='small'
            bordered
            columns={columns}
            dataSource={rows}
            expandable={{
              defaultExpandAllRows: false,
            }}
          />
        )}

        {moduleComponent()}

        {flameData && flameData.length > 0 && (
          <FlameGraph
            data={flameData[module]}
            height={flameHeight}
            width={cardWidth}
            onChange={(node: any): void => {}}
          />
        )}

        <div ref={timelineRef} />
      </Card>
    </div>
  );
};
