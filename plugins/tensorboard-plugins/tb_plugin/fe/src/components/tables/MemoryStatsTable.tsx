/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';
import { Table } from 'antd';
import { makeStyles } from '@material-ui/core';

export interface IProps {
  data: any;
  sort: string;
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap',
  },
}));

const getMemoryStatsTableColumns = function (columns: any, sort: string, tooltipClass: string): any {
  let i = 0;
  return columns.map((col: any) => {
    const key = `col${i++}`;
    const stringCompare = (a: any, b: any): number => a[key].localeCompare(b[key]);
    const numberCompare = (a: any, b: any): number => (a[key] || 0) - (b[key] || 0);
    return {
      dataIndex: key,
      key: key,
      title: col.name,
      sorter: col.type === 'string' ? stringCompare : numberCompare,
      defaultSortOrder: sort === col.name ? ('descend' as const) : undefined,
      showSorterTooltip: col.tooltip ? { title: col.tooltip, overlayClassName: tooltipClass } : true,
    };
  });
};

const getMemoryStatsTableRows = function (rows: any): any {
  return rows.map((row: any) => {
    let i = 0;
    const res: any = {};
    row.forEach((entry: any) => {
      res[`col${i++}`] = entry;
    });
    return res;
  });
};

export const MemoryStatsTable = (props: IProps): React.JSX.Element => {
  const { data, sort } = props;
  const classes = useStyles();

  const rows = React.useMemo(() => getMemoryStatsTableRows(data.rows), [data.rows]);

  const columns = React.useMemo(
    () => getMemoryStatsTableColumns(data.columns, sort, classes.tooltip),
    [data.columns, sort, classes.tooltip]
  );

  const [pageSize, setPageSize] = React.useState(30);
  const onShowSizeChange = (current: number, size: number): void => {
    setPageSize(size);
  };

  return (
    <Table
      size='small'
      bordered
      columns={columns}
      dataSource={rows}
      pagination={{
        pageSize,
        pageSizeOptions: ['10', '20', '30', '50', '100'],
        onShowSizeChange,
      }}
    />
  );
};
