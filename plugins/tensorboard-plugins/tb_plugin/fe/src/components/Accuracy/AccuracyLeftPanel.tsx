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

import * as React from 'react'
import { useState, useEffect, useCallback, useRef } from 'react'
import { makeStyles } from '@material-ui/core/styles'
import { Button, Checkbox, Spin, Modal, message } from 'antd'
import { CheckboxChangeEvent } from 'antd/es/checkbox'
import {
  DeleteOutlined,
  DownloadOutlined,
  ImportOutlined,
  SettingOutlined,
  WarningTwoTone,
} from '@ant-design/icons'
import { RegexConfigModal } from './RegexConfigModal'
import { FileInfo } from './entity'

interface IProps {
  onChangeCheckedFileList: (files: FileInfo[]) => void
  onChangeUploadedCount: (count: number) => void
}

// 匹配数字包括科学计数法
const LOSS_REG_EXP = /[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/
// 匹配自然数
const ITER_REG_EXP = /\d+/
// 单个文件最大大小
const FILE_MAX_SIZE = 50 * 1024 * 1024
// 最大文件上传数量
export const MAX_FILE_COUNT = 6

const useStyles = makeStyles(() => ({
  root: {
    height: '100%'
  },
  btnPanel: {
    height: 50,
    lineHeight: '50px',
    borderBottom: '1px solid #DFE5EF',
    display: 'flex',
    '& .ant-btn': {
      margin: 'auto'
    }
  },
  fileContainer: {
    height: 54,
    padding: '0 24px',
    display: 'flex',
    alignItems: 'center',
    '& .fileNameLabel': {
      display: 'inline-block',
      marginLeft: 12,
      width: 200,
      fontSize: 14,
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap'
    },
    '& .btns': {
      display: 'inline-block',
      marginLeft: 'auto',
      '& .icon': {
        cursor: 'pointer',
        '&:hover': {
          color: '#1890ff'
        }
      },
      '& .iconLeft': {
        marginRight: 8
      }
    },
  },
  deleteModal: {
    '& .ant-modal-title': {
      fontWeight: 'bold'
    },
    '& .deleteModalBody': {
      display: 'flex',
      alignItems: 'center',
      height: 80,
      '& .warningIcon': {
        display: 'inline-block',
        fontSize: 50
      },
      '& .warningText': {
        display: 'inline-block',
        marginLeft: 16,
        overflow: 'hidden',
        wordBreak: 'break-all',
        flex: 1
      }
    }
  }
}))

export const AccuracyLeftPanel: React.FC<IProps> = (props) => {
  const { onChangeCheckedFileList, onChangeUploadedCount } = props
  const classes = useStyles()
  const [configModalVis, setConfigModalVis] = useState<boolean>(false)
  const [deleteModalVis, setDeleteModalVis] = useState<boolean>(false)
  const [fileList, setFileList] = useState<FileInfo[]>([])
  const [importSpin, setImportSpin] = useState<boolean>(false)
  const [selectedFile, setSelectedFile] = useState<FileInfo | undefined>(undefined)
  const downLoadRef = useRef<HTMLAnchorElement>(null)

  const parseFile = (file: FileInfo): FileInfo => {
    file.losses = []
    file.iterLosses = {}
    file.iters = []
    const lines = file.fileContent.split(/\r\n|\n|\r/)
    for (let i = 0; i < lines.length; i++) {
      const iter = parseByTag(lines[i], file.iterTag, false)
      const loss = parseByTag(lines[i], file.lossTag, true)
      if (iter !== null && loss !== null) {
        file.iters.push(iter)
        file.losses.push([iter, loss])
        file.iterLosses[iter] = loss
      }
    }
    return file
  }

  const parseByTag = (line: string, tag: string, isLoss: boolean): number | null => {
    let pos = line.indexOf(tag)
    let result: number | null = null
    if (pos !== -1) {
      const res = (isLoss ? LOSS_REG_EXP : ITER_REG_EXP)
        .exec(line.substring(pos + tag.length).trim().split(/\s+/)[0])
      if (res !== null) {
        if (isLoss) {
          result = parseFloat(res[0])
        } else {
          result = parseInt(res[0])
        }
      } else {
        console.log(`Found ${isLoss ? 'loss' : 'iteration'} text, but parse value with error: [${line}]`)
      }
    }
    return result
  }

  const importFile = () => {
    document.getElementById('accComparisonSelectFile')?.click()
  }

  const uploadFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    setImportSpin(true)
    const file = e.target.files?.[0]
    if (file) {
      if (file.size > FILE_MAX_SIZE) {
        message.warn('Sorry, the file size cannot be greater than 50MB.')
        setImportSpin(false)
        // 防止同名文件不触发事件
        e.target.value = ''
        return
      }
      const reader = new FileReader()
      reader.onload = ((selectedFile) => {
        return (e) => {
          addFile(selectedFile.name.trim(), e.target?.result as string)
          setImportSpin(false)
        }
      })(file);
      reader.readAsText(file)
    }
    // 防止同名文件不触发事件
    e.target.value = ''
  }

  const addFile = (fileName: string, fileContent: string) => {
    // 限制文件后缀为.log或.txt
    const fileLength = fileName.length
    if (fileLength <= 4 || !['.txt', '.log'].includes(fileName.slice(fileLength - 4).toLowerCase())) {
      message.warn('Please select a file with the extension of "txt" or "log"')
      return
    }
    const tempList: FileInfo[] = JSON.parse(JSON.stringify(fileList))
    // 上传同名文件加上(1~最大文件数减1)标识
    if (!!tempList.find(item => item.fileName === fileName)) {
      for (let i = 1; i < MAX_FILE_COUNT; i++) {
        let temp = `${fileName.slice(0, fileLength - 4)}(${i})${fileName.slice(fileLength - 4)}`
        if (tempList.find(item => item.fileName === temp) === undefined) {
          fileName = temp
          break
        }
      }
    }
    const file: FileInfo = {
      id: fileList.length,
      fileName: fileName,
      fileContent,
      checked: true,
      lossTag: 'loss:',
      iterTag: 'iteration',
      iters: [],
      losses: [],
      iterLosses: {}
    }
    tempList.push(parseFile(file))
    setFileList(tempList)
  }

  const exportCsv = (data: FileInfo) => {
    let csvContent = `data:text/csv;charset=utf-8,${data.iterTag},${data.lossTag}\n`
    data.losses.forEach(item => {
      csvContent += `${item[0]},${item[1]}\n`
    })
    downLoadRef.current?.setAttribute('href', encodeURI(csvContent))
    downLoadRef.current?.setAttribute('download', `${data.fileName}.csv`)
    downLoadRef.current?.click()
  }

  const onCheckChange = (e: CheckboxChangeEvent, index: number) => {
    const tempList: FileInfo[] = JSON.parse(JSON.stringify(fileList))
    tempList[index].checked = e.target.checked
    setFileList(tempList)
  }

  const onConfigIconClick = (data: FileInfo) => {
    setSelectedFile(data)
    setConfigModalVis(true)
  }

  const onDeleteIconClick = (data: FileInfo) => {
    setSelectedFile(data)
    setDeleteModalVis(true)
  }

  const configModalOk = (data: FileInfo) => {
    const tempList = fileList.map(item => {
      return item.id === data.id ? parseFile(data) : item
    })
    setFileList(tempList)
    setConfigModalVis(false)
  }

  const configModalCancel = () => {
    setConfigModalVis(false)
  }

  const deleteModalOk = () => {
    const tempList = JSON.parse(JSON.stringify(fileList))
    let founded = false
    let index = 0
    for (let i = 0; i < tempList.length; i++) {
      if (founded) {
        tempList[i].id -= 1
        continue
      }
      if (tempList[i].id === selectedFile?.id) {
        founded = true
        index = i
      }
    }
    tempList.splice(index, 1)
    setFileList(tempList)
    setSelectedFile(undefined)
    setDeleteModalVis(false)
  }

  const renderFileItems = useCallback(() => {
    return fileList.map((item) => {
      return (
        <div key={item.id} className={classes.fileContainer}>
          <Checkbox checked={item.checked} onChange={(e) => onCheckChange(e, item.id)} />
          <span className="fileNameLabel" title={item.fileName}>{item.fileName}</span>
          <div className="btns">
            <SettingOutlined className="icon iconLeft" title="Config" onClick={() => onConfigIconClick(item)} />
            <DownloadOutlined className="icon iconLeft" title='Export' onClick={() => exportCsv(item)} />
            <DeleteOutlined className="icon" title='Delete' onClick={() => onDeleteIconClick(item)} />
          </div>
        </div>
      )
    })
  }, [JSON.stringify(fileList)])

  useEffect(() => {
    onChangeCheckedFileList(fileList.filter(item => item.checked))
    onChangeUploadedCount(fileList.length)
  }, [JSON.stringify(fileList)])

  return (
    <div className={classes.root}>
      <Spin spinning={importSpin} tip="importing...">
        <div className={classes.btnPanel}>
          <Button
            icon={<ImportOutlined />}
            onClick={importFile}
            disabled={fileList.length >= MAX_FILE_COUNT}
            title={`You can import no more than ${MAX_FILE_COUNT} files.`}
          >
            Import files
          </Button>
          <input
            id='accComparisonSelectFile'
            style={{ display: 'none' }}
            type='file'
            accept='.txt,.log'
            onChange={uploadFile}
          />
        </div>
        {renderFileItems()}
      </Spin>
      {configModalVis &&
        <RegexConfigModal
          file={selectedFile as FileInfo}
          onOk={configModalOk}
          onCancel={configModalCancel}
        />
      }
      <Modal
        title='Delete reminder'
        open={deleteModalVis}
        centered
        maskClosable={false}
        onCancel={() => setDeleteModalVis(false)}
        onOk={deleteModalOk}
        width={500}
        className={classes.deleteModal}
      >
        <div className="deleteModalBody">
          <WarningTwoTone className="warningIcon" twoToneColor="rgb(252, 197, 96)" />
          <span className="warningText" title={selectedFile?.fileName}>
            Are you sure to delete "<b>{selectedFile?.fileName}</b>"?
          </span>
        </div>
      </Modal>
      <a ref={downLoadRef} />
    </div>
  )
}
