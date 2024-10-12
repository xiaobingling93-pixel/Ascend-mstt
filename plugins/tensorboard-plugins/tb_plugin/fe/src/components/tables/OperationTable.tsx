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

import * as React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import {
  OperationTableData,
  OperationTableDataInner,
  TableMetadata,
} from '../../api';
import { OperationGroupBy } from '../../constants/groupBy';
import { attachId, getCommonOperationColumns } from './common';
import { Table, TablePaginationConfig, TableProps } from 'antd';
import { makeExpandIcon } from './ExpandIcon';
import { CallStackTable } from './CallStackTable';

export interface IProps {
  data: OperationTableData;
  run: string;
  worker: string;
  span: string;
  groupBy: OperationGroupBy;
  sortColumn: string;
  tooltips?: any;
  deviceTarget: string;
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap',
  },
}));

const rowExpandable = (record: OperationTableDataInner) =>
  record.has_call_stack;
const expandIcon = makeExpandIcon<OperationTableDataInner>(
  'View CallStack',
  (record) => !record.has_call_stack
);
export const OperationTable = (props: IProps) => {
  const {
    data,
    run,
    worker,
    span,
    groupBy,
    sortColumn,
    tooltips,
    deviceTarget,
  } = props;
  const classes = useStyles(props);

  const rows = React.useMemo(() => attachId(data), [data]);

  const columns = React.useMemo(
    () =>
      getCommonOperationColumns(
        rows,
        deviceTarget,
        sortColumn,
        tooltips,
        classes
      ),
    [rows]
  );

  const [pageSize, setPageSize] = React.useState(30);
  const onShowSizeChange = (current: number, size: number) => {
    setPageSize(size);
  };

  const expandIconColumnIndex = columns.length;
  const expandedRowRender = React.useCallback(
    (record: OperationTableDataInner) => (
      <CallStackTable
        data={record}
        run={run}
        worker={worker}
        span={span}
        groupBy={groupBy}
        deviceTarget={deviceTarget}
      />
    ),
    [run, worker, span, groupBy]
  );

  const expandable: TableProps<OperationTableDataInner>['expandable'] =
    React.useMemo(
      () => ({
        expandIconColumnIndex,
        expandIcon,
        expandedRowRender,
        rowExpandable,
      }),
      [expandIconColumnIndex, expandedRowRender]
    );

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
      expandable={expandable}
    />
  );
};
