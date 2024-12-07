/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles';
import * as React from 'react';

export interface IProps {
  gpuInfo: any;
}

const useStyles = makeStyles((theme) => ({
  root: {
    border: '1px solid #E0E0E0',
    borderCollapse: 'collapse',
    width: '100%',
  },
  td: {
    borderTop: '1px solid #E0E0E0',
    borderBottom: '1px solid #E0E0E0',
    borderCollapse: 'collapse',
    paddingLeft: 10,
    paddingRight: 10,
  },
  nodeTd: {
    fontWeight: 'bold',
  },
  pidTd: {
    fontWeight: 'normal',
  },
  gpuTd: {
    fontWeight: 'normal',
  },
  keyTd: {
    fontWeight: 'normal',
    textAlign: 'right',
  },
  valueTd: {
    fontWeight: 'bold',
  },
}));

interface TableCellInfo {
  content: string;
  rowspan: number;
  cellType: 'node' | 'pid' | 'gpu' | 'key' | 'value';
  last?: boolean;
}

function makeTableCellInfo(gpuInfo: any): TableCellInfo[][] {
  const rows: TableCellInfo[][] = [];
  let curr_row: TableCellInfo[] = [];
  rows.push(curr_row);
  Object.keys(gpuInfo.data).forEach((node_name) => {
    const node_cell = {
      content: node_name,
      rowspan: 0,
      cellType: 'node' as const,
    };
    const i = rows.length;
    curr_row.push(node_cell);
    Object.keys(gpuInfo.data[node_name]).forEach((pid) => {
      const pid_cell = { content: pid, rowspan: 0, cellType: 'pid' as const };
      const j = rows.length;
      curr_row.push(pid_cell);
      Object.keys(gpuInfo.data[node_name][pid]).forEach((gpu) => {
        const gpu_cell = { content: gpu, rowspan: 0, cellType: 'gpu' as const };
        const k = rows.length;
        curr_row.push(gpu_cell);
        Object.keys(gpuInfo.data[node_name][pid][gpu]).forEach((key_name) => {
          curr_row.push({
            content: key_name,
            rowspan: 1,
            cellType: 'key' as const,
          });
          const value: string = gpuInfo.data[node_name][pid][gpu][key_name];
          curr_row.push({
            content: value,
            rowspan: 1,
            cellType: 'value' as const,
          });
          curr_row = [];
          rows.push(curr_row);
        });
        gpu_cell.rowspan = rows.length - k;
      });
      pid_cell.rowspan = rows.length - j;
    });
    node_cell.rowspan = rows.length - i;
  });
  rows.pop();
  return rows;
}

export const GpuInfoTable: React.FC<IProps> = (props) => {
  const classes = useStyles();
  interface TableCellInfoNoLast {
    content: string;
    rowspan: number;
    cellType: 'node' | 'pid' | 'gpu' | 'key' | 'value';
  }

  const rows = React.useMemo(
    () => makeTableCellInfo(props.gpuInfo),
    [props.gpuInfo]
  );

  const cellToClass = {
    node: classes.nodeTd,
    pid: classes.pidTd,
    gpu: classes.gpuTd,
    key: classes.keyTd,
    value: classes.valueTd,
  };

  const renderCell = function (info: TableCellInfoNoLast) {
    let cellClass = cellToClass[info.cellType];
    let content = info.cellType === 'key' ? `${info.content}:` : info.content;
    return (
      <td className={`${classes.td} ${cellClass}`} rowSpan={info.rowspan}>
        {content}
      </td>
    );
  };

  return (
    <table className={classes.root}>
      {rows.map((row) => (
        <tr>{row.map(renderCell)}</tr>
      ))}
    </table>
  );
};
