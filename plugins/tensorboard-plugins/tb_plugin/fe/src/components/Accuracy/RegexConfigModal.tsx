/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------
 * Copyright (c) 2023, Huawei Technologies.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0  (the 'License')
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *--------------------------------------------------------------------------------------------*/

import { Input, message, Modal } from 'antd';
import * as React from 'react';
import { useState } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { FileInfo } from './entity';

interface IProps {
  file: FileInfo;
  onOk: (file: FileInfo) => void;
  onCancel: () => void;
}

const useStyles = makeStyles(() => ({
  root: {
    '& .ant-modal-title': {
      width: 500,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      fontWeight: 'bold',
    },
  },
  filterItem: {
    display: 'flex',
    height: 56,
    width: '100%',
    alignItems: 'center',
    '& .tagLabel': {
      display: 'inline-block',
      width: 100,
    },
    '& .ant-input': {
      width: 320,
      height: 32,
    },
    '& .ant-checkbox-wrapper': {
      marginLeft: 'auto',
    },
  },
}));

export const RegexConfigModal: React.FC<IProps> = (props) => {
  const classes = useStyles();
  const [lossTag, setLossTag] = useState<string>(props.file.lossTag);
  const [iterTag, setIterTag] = useState<string>(props.file.iterTag);

  const lossTagChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setLossTag(e.target.value);
  };

  const iterTagChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIterTag(e.target.value);
  };

  const configModalOk = () => {
    if (lossTag.trim() === '') {
      message.warning('Loss Tag cannot be empty or only spaces!');
      return;
    }
    if (iterTag.trim() === '') {
      message.warning('Iteration Tag cannot be empty or only spaces!');
      return;
    }
    if (lossTag === props.file.lossTag && iterTag === props.file.iterTag) {
      props.onCancel();
    } else {
      const configFile: FileInfo = {
        ...props.file,
        lossTag,
        iterTag,
      };
      props.onOk(configFile);
    }
  };

  return (
    <Modal
      title={props.file.fileName}
      onOk={configModalOk}
      onCancel={props.onCancel}
      open
      centered
      maskClosable={false}
      width={480}
      className={classes.root}
    >
      <div className={classes.filterItem}>
        <span className='tagLabel'>Loss Tag</span>
        <Input onChange={lossTagChange} value={lossTag} maxLength={200} />
      </div>
      <div className={classes.filterItem}>
        <span className='tagLabel'>Iteration Tag</span>
        <Input onChange={iterTagChange} value={iterTag} maxLength={200} />
      </div>
    </Modal>
  );
};
