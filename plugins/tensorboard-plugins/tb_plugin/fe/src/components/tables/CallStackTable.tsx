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
import { CallStackTableData, OperationTableDataInner } from '../../api';
import { Table, TableProps } from 'antd';

import * as api from '../../api';
import { transformTableData, TransformedCallStackDataInner } from './transform';
import { attachId, getCommonOperationColumns } from './common';
import { OperationGroupBy } from '../../constants/groupBy';
import { makeExpandIcon } from './ExpandIcon';
import { CallFrameList } from './CallFrameList';

export interface IProps {
  data: OperationTableDataInner;
  run: string;
  worker: string;
  span: string;
  groupBy: OperationGroupBy;
  deviceTarget: string;
}

const useStyles = makeStyles((theme) => ({
  tooltip: {
    whiteSpace: 'pre-wrap',
  },
}));

const expandIcon = makeExpandIcon<TransformedCallStackDataInner>(
  'View call frames',
  (record) => !record.callStackFrames.length
);

const rowExpandable = (record: TransformedCallStackDataInner) =>
  !!record.callStackFrames.length;
const expandedRowRender = (record: TransformedCallStackDataInner) => (
  <CallFrameList callFrames={record.callStackFrames} />
);

export const CallStackTable = (props: IProps) => {
  const { data, run, worker, span, groupBy, deviceTarget } = props;
  const { name, input_shape } = data;
  const classes = useStyles(props);

  const [stackData, setStackData] = React.useState<
    CallStackTableData | undefined
  >(undefined);
  const [tooltips, setTooltips] = React.useState<any | undefined>();

  React.useEffect(() => {
    api.defaultApi
      .operationStackGet(run, worker, span, groupBy, name, input_shape)
      .then((resp) => {
        setTooltips(resp.metadata.tooltips);
        setStackData(resp.data);
      });
  }, [name, input_shape, run, worker, span, groupBy]);

  const transformedData = React.useMemo(
    () => stackData && transformTableData(attachId(stackData)),
    [stackData]
  );

  const columns = React.useMemo(
    () =>
      transformedData &&
      getCommonOperationColumns(
        transformedData,
        deviceTarget,
        undefined,
        tooltips,
        classes
      ),
    [transformedData]
  );

  const expandIconColumnIndex = columns?.length;

  const expandable: TableProps<TransformedCallStackDataInner>['expandable'] =
    React.useMemo(
      () => ({
        expandIconColumnIndex,
        expandIcon,
        expandedRowRender,
        rowExpandable,
      }),
      [expandIconColumnIndex]
    );

  return (
    <Table
      loading={!transformedData}
      size='small'
      bordered
      columns={columns}
      dataSource={transformedData}
      expandable={expandable}
    />
  );
};
