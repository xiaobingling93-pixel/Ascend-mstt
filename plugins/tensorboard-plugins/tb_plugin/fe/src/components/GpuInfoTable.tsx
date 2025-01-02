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
  let currRow: TableCellInfo[] = [];
  rows.push(currRow);
  Object.keys(gpuInfo.data).forEach((nodeName) => {
    const nodeCell = {
      content: nodeName,
      rowspan: 0,
      cellType: 'node' as const,
    };
    const i = rows.length;
    currRow.push(nodeCell);
    Object.keys(gpuInfo.data[nodeName]).forEach((pid) => {
      const pidCell = { content: pid, rowspan: 0, cellType: 'pid' as const };
      const j = rows.length;
      currRow.push(pidCell);
      Object.keys(gpuInfo.data[nodeName][pid]).forEach((gpu) => {
        const gpuCell = { content: gpu, rowspan: 0, cellType: 'gpu' as const };
        const k = rows.length;
        currRow.push(gpuCell);
        Object.keys(gpuInfo.data[nodeName][pid][gpu]).forEach((keyName) => {
          currRow.push({
            content: keyName,
            rowspan: 1,
            cellType: 'key' as const,
          });
          const value: string = gpuInfo.data[nodeName][pid][gpu][keyName];
          currRow.push({
            content: value,
            rowspan: 1,
            cellType: 'value' as const,
          });
          currRow = [];
          rows.push(currRow);
        });
        gpuCell.rowspan = rows.length - k;
      });
      pidCell.rowspan = rows.length - j;
    });
    nodeCell.rowspan = rows.length - i;
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

  const rows = React.useMemo(() => makeTableCellInfo(props.gpuInfo), [props.gpuInfo]);

  const cellToClass = {
    node: classes.nodeTd,
    pid: classes.pidTd,
    gpu: classes.gpuTd,
    key: classes.keyTd,
    value: classes.valueTd,
  };

  const renderCell = function (info: TableCellInfoNoLast): JSX.Element {
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
