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
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { FileInfo } from './entity';
import { Empty, message } from 'antd';
import { LossDisplayPanel } from './LossDisplayPanel';
import { ComparisonPanel } from './ComparisonPanel';
import { MAX_FILE_COUNT } from './AccuracyLeftPanel';

interface IProps {
  fileList: FileInfo[];
  fileCount: number;
}

const useStyles = makeStyles(() => ({
  root: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    backgroundColor: 'white',
    height: '100%',
    overflowY: 'auto',
    '& .welcomeLabel': {
      marginTop: '18%',
      font: '36px bold',
    },
    '& .importText': {
      fontSize: 20,
      fontWeight: 400,
      margin: '24px 0',
      '& span': {
        cursor: 'pointer',
        color: '#0077ff',
      },
    },
  },
}));

export const LossComparison: React.FC<IProps> = (props) => {
  const { fileList, fileCount } = props;
  const classes = useStyles();

  const onImportFile = () => {
    if (fileCount >= MAX_FILE_COUNT) {
      message.warn(`You can import no more than ${MAX_FILE_COUNT} files.`);
      return;
    }
    document.getElementById('accComparisonSelectFile')?.click();
  };

  return (
    <div className={classes.root}>
      {fileList.length <= 0 ? (
        <>
          <span className='welcomeLabel'>Welcome to loss comparison</span>
          <div className='importText'>
            Select left files or{' '}
            <span onClick={onImportFile}>Import files</span>
          </div>
          <Empty description={false} />
        </>
      ) : (
        <>
          <LossDisplayPanel fileList={fileList} />
          {fileList.length > 1 && <ComparisonPanel fileList={fileList} />}
        </>
      )}
    </div>
  );
};
