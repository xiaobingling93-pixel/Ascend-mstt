/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Button from '@material-ui/core/Button'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'
import Typography from '@material-ui/core/Typography'
import ChevronLeftIcon from '@material-ui/icons/ChevronLeft'
import { Select, Table } from 'antd'
import * as React from 'react'
import * as api from '../api'
import { useResizeEventDependency } from '../utils/resize'
import { FullCircularProgress } from './FullCircularProgress'
import * as echarts from 'echarts'

const { Option } = Select

const topGraphHeight = 230

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  },
  pre: {
    '& ul': {
      margin: 0,
      paddingLeft: theme.spacing(3),
      ...theme.typography.body1
    },
    '& li': {},
    '& a': {
      color: '#ffa726'
    },
    '& a:active': {
      color: '#ffa726'
    },
    '& p': {
      margin: 0,
      ...theme.typography.subtitle1,
      fontWeight: theme.typography.fontWeightBold
    }
  },
  topGraph: {
    height: topGraphHeight + 40
  },
  iconButton: {
    padding: '8px'
  }
}))

const getAngleByDataLength = (data: number) => {
  if (data < 10) {
    return 0
  } else {
    // 数量越大越趋近于旋转90度
    return 90 * (1 - 10 / data)
  }
}

export interface DiffColumnChartIProps {
  rawData: any[]
  selectCallback: (row: number, column: number) => void
}

export interface DiffStepChartIProps {
  rawData: any[]
}

const DiffColumnChart: React.FC<DiffColumnChartIProps> = (
  props: DiffColumnChartIProps
) => {
  const { rawData, selectCallback } = props
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    let left_duration_data: number[] = []
    let left_accumulated_duration_data: number[] = []

    let right_duration_data: number[] = []
    let right_accumulated_duration_data: number[] = []

    for (let i = 0; i < rawData.length; i++) {
      let curr = rawData[i]
      left_duration_data.push(curr[1])
      right_duration_data.push(curr[2])
      left_accumulated_duration_data.push(curr[3])
      right_accumulated_duration_data.push(curr[4])
    }

    let left_duration_max = Math.max(...left_duration_data)
    let right_duration_max = Math.max(...right_duration_data)
    let duration_max = Math.max(left_duration_max, right_duration_max)

    let left_accumulated_duration_max = Math.max(
      ...left_accumulated_duration_data
    )
    let right_accumulated_duration_max = Math.max(
      ...right_accumulated_duration_data
    )
    let accumulated_max = Math.max(
      left_accumulated_duration_max,
      right_accumulated_duration_max
    )

    const chart = echarts.init(element)

    const options: echarts.EChartsOption = {
      title: {
        text: 'Execution Comparsion'
      },
      legend: {
        top: 10,
        right: 10
      },
      tooltip: {
        trigger: 'axis',
        formatter: function (params: any) {
          const index = params[0].name.indexOf('@')
          const safeName = params[0].name.replace(/</g, '&lt;').replace(/>/g, '&gt;')
          var res = `<b>${index > -1 ? safeName.slice(index + 1) : safeName}<b/> <br/>`
          for (const item of params) {
            if (typeof item.value[item.encode.y[0]] === 'number') {
              res += `<span style="background: ${item.color};
              height:10px;
              width: 10px;
              border-radius: 50%;
              display: inline-block;
              margin-right:10px;">
              </span>
              ${item.seriesName}: ${item.value[item.encode.y[0]]}<br/>`
            }
          }
          return res
        }
      },
      series: [
        {
          type: 'bar',
          itemStyle: {
            color: '#3366cc'
          },
          yAxisIndex: 0,

        },
        {
          type: 'bar',
          itemStyle: {
            color: '#dc3912'
          },
          yAxisIndex: 0
        },
        {
          type: 'line',
          itemStyle: {
            color: '#ff9900'
          },
          yAxisIndex: 1
        },
        {
          type: 'line',
          itemStyle: {
            color: '#109618'
          },
          yAxisIndex: 1
        }
      ],
      xAxis: {
        type: 'category',
        axisLabel: {
          interval: 0,
          rotate: getAngleByDataLength(rawData.length),
          formatter: (name: string) => {
            const index = name.indexOf('@')
            if (index > -1) {
              name = name.slice(index + 1)
            }
            return name.length > 16 ? name.slice(0, 14) + "..." : name;
          }
        }
      },
      yAxis: [{
        type: 'value',
        name: 'Time Difference(us)',
        scale: true
      }, {
        type: 'value',
        name: 'Accumulated Difference(us)',
        scale: true
      }],
      dataset: {
        source: rawData.map((item, idx) => {
          // 添加索引保证x轴刻度不重复
          let param: any[] = [...item]
          param[0] = `${idx}@${param[0]}`
          return param
        })
      }
    }

    options && chart.setOption(options, true)
    chart.on('click', (param) => {
      if (param.seriesIndex !== undefined) {
        selectCallback(param.dataIndex, param.seriesIndex + 1)
      }
    })

    return () => {
      chart.dispose()
    }
  }, [rawData, resizeEventDependency])

  return (
    <div>
      <div ref={graphRef} style={{ height: '400px' }}></div>
    </div>
  )
}

const DiffStepChart: React.FC<DiffStepChartIProps> = (
  props: DiffStepChartIProps
) => {
  const { rawData } = props
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return
    const chart = echarts.init(element)
    const options: echarts.EChartsOption = {
      title: {
        text: 'Execution Diff'
      },
      legend: {
        top: 10,
        right: 10
      },
      dataset: {
        source: rawData.map((item, idx) => {
          // 添加索引保证x轴刻度不重复
          let param: any[] = [...item]
          param[0] = `${idx}@${param[0]}`
          return param
        })
      },
      xAxis: {
        type: 'category',
        axisLabel: {
          interval: 0,
          rotate: getAngleByDataLength(rawData.length),
          formatter: (name: string) => {
            const index = name.indexOf('@')
            if (index > -1) {
              name = name.slice(index + 1)
            }
            return name.length > 16 ? name.slice(0, 14) + "..." : name;
          }
        }
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      tooltip: {
        trigger: 'axis',
        formatter: function (params: any) {
          const index = params[0].name.indexOf('@')
          const safeName = params[0].name.replace(/</g, '&lt;').replace(/>/g, '&gt;')
          var res = `<b>${index > -1 ? safeName.slice(index + 1) : safeName}<b/> <br/>`
          for (const item of params) {
            if (typeof item.value[item.encode.y[0]] === 'number') {
              res += `<span style="background: ${item.color};
              height:10px;
              width: 10px;
              border-radius: 50%;
              display: inline-block;
              margin-right:10px;">
              </span>
              ${item.seriesName}: ${item.value[item.encode.y[0]]}<br/>`
            }
          }
          return res
        }
      },
      series: [
        {
          type: 'line',
          color: '#3366cc',
          symbolSize: 0,
          step: 'middle',
          areaStyle: {
            color: '#c1d1ef',
            opacity: 1
          }
        }, {
          type: 'line',
          color: '#dc3912',
          symbolSize: 0,
          step: 'middle',
          areaStyle: {
            color: '#f4c3b7',
            opacity: 1
          }
        }
      ]
    }

    options && chart.setOption(options, true)
    return () => {
      chart.dispose()
    }
  }, [rawData, resizeEventDependency])

  return (
    <div>
      <div ref={graphRef} style={{ height: 500 }}></div>
    </div>
  )
}

export interface IProps {
  run: string
  worker: string
  span: string
  expRun: string
  expWorker: string
  expSpan: string
}

export interface ColumnUnderlyingData {
  name: string
  path: string
  leftAggs: any[]
  rightAggs: any[]
}

export interface TableRow {
  key: number

  operator: string
  baselineCalls?: number
  expCalls?: number
  deltaCalls?: number
  deltaCallsPercentNumber?: number
  deltaCallsPercent?: string

  baselineHostDuration: number
  expHostDuration: number
  deltaHostDuration: number
  deltaHostDurationPercentNumber: number
  deltaHostDurationPercent: string

  baselineSelfHostDuration: number
  expSelfHostDuration: number
  deltaSelfHostDuration: number
  deltaSelfHostDurationPercentNumber: number
  deltaSelfHostDurationPercent: string

  baselineDeviceDuration: number
  expDeviceDuration: number
  deltaDeviceDuration: number
  deltaDeviceDurationPercentNumber: number
  deltaDeviceDurationPercent: string

  baselineSelfDeviceDuration: number
  expSelfDeviceDuration: number
  deltaSelfDeviceDuration: number
  deltaSelfDeviceDurationPercentNumber: number
  deltaSelfDeviceDurationPercent: string
}

let columnChartDataStack: any[][] = []
let stepChartDataStack: any[][] = []
let columnUnderlyingDataStack: ColumnUnderlyingData[][] = []
let columnTableDataSourceStack: TableRow[][] = []

export const DiffOverview: React.FC<IProps> = (props: IProps) => {
  // #region - Constant

  const COMPOSITE_NODES_NAME = 'CompositeNodes'

  const hostDurationColumns = [
    {
      title: 'Baseline Host Duration (us)',
      dataIndex: 'baselineHostDuration',
      key: 'baselineHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineHostDuration - b.baselineHostDuration
    },
    {
      title: 'Exp Host Duration (us)',
      dataIndex: 'expHostDuration',
      key: 'expHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expHostDuration - b.expHostDuration
    },
    {
      title: 'Delta Host Duration (us)',
      dataIndex: 'deltaHostDuration',
      key: 'deltaHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaHostDuration! - b.deltaHostDuration!
    },
    {
      title: 'Delta Host Duration%',
      dataIndex: 'deltaHostDurationPercent',
      key: 'deltaHostDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaHostDurationPercentNumber! - b.deltaHostDurationPercentNumber!
    }
  ]

  const selfHostDurationColumns = [
    {
      title: 'Baseline Self Host Duration (us)',
      dataIndex: 'baselineSelfHostDuration',
      key: 'baselineSelfHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineSelfHostDuration - b.baselineSelfHostDuration
    },
    {
      title: 'Exp Self Host Duration (us)',
      dataIndex: 'expSelfHostDuration',
      key: 'expSelfHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expSelfHostDuration - b.expSelfHostDuration
    },
    {
      title: 'Delta Self Host Duration (us)',
      dataIndex: 'deltaSelfHostDuration',
      key: 'deltaSelfHostDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfHostDuration! - b.deltaSelfHostDuration!
    },
    {
      title: 'Delta Self Host Duration%',
      dataIndex: 'deltaSelfHostDurationPercent',
      key: 'deltaSelfHostDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfHostDurationPercentNumber! -
        b.deltaSelfHostDurationPercentNumber!
    }
  ]

  const deviceDurationColumns = [
    {
      title: 'Baseline Device Duration (us)',
      dataIndex: 'baselineDeviceDuration',
      key: 'baselineDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineDeviceDuration - b.baselineDeviceDuration
    },
    {
      title: 'Exp Device Duration (us)',
      dataIndex: 'expDeviceDuration',
      key: 'expDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expDeviceDuration - b.expDeviceDuration
    },
    {
      title: 'Delta Device Duration (us)',
      dataIndex: 'deltaDeviceDuration',
      key: 'deltaDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaDeviceDuration! - b.deltaDeviceDuration!
    },
    {
      title: 'Delta Device Duration%',
      dataIndex: 'deltaDeviceDurationPercent',
      key: 'deltaDeviceDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaDeviceDurationPercentNumber! -
        b.deltaDeviceDurationPercentNumber!
    }
  ]

  const selfDeviceDurationColumns = [
    {
      title: 'Baseline Self Device Duration (us)',
      dataIndex: 'baselineSelfDeviceDuration',
      key: 'baselineSelfDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.baselineSelfDeviceDuration - b.baselineSelfDeviceDuration
    },
    {
      title: 'Exp Self Device Duration (us)',
      dataIndex: 'expSelfDeviceDuration',
      key: 'expSelfDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.expSelfDeviceDuration - b.expSelfDeviceDuration
    },
    {
      title: 'Delta Self Device Duration (us)',
      dataIndex: 'deltaSelfDeviceDuration',
      key: 'deltaSelfDeviceDuration',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfDeviceDuration! - b.deltaSelfDeviceDuration!
    },
    {
      title: 'Delta Self Device Duration%',
      dataIndex: 'deltaSelfDeviceDurationPercent',
      key: 'deltaSelfDeviceDurationPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaSelfDeviceDurationPercentNumber! -
        b.deltaSelfDeviceDurationPercentNumber!
    }
  ]

  type IColumnMapType = { [key: string]: any }

  const tableSourceColumnMap: IColumnMapType = {
    selfHostDuration: selfHostDurationColumns,
    hostDuration: hostDurationColumns,
    deviceDuration: deviceDurationColumns,
    selfDeviceDuration: selfDeviceDurationColumns
  }

  const baseTableColumns = [
    {
      title: 'Operator',
      dataIndex: 'operator',
      key: 'operator',
      sorter: (a: TableRow, b: TableRow) => a.operator.localeCompare(b.operator)
    },
    {
      title: 'Baseline Calls',
      dataIndex: 'baselineCalls',
      key: 'baselineCalls',
      sorter: (a: TableRow, b: TableRow) => a.baselineCalls! - b.baselineCalls!
    },
    {
      title: 'Exp Calls',
      dataIndex: 'expCalls',
      key: 'expCalls',
      sorter: (a: TableRow, b: TableRow) => a.expCalls! - b.expCalls!
    },
    {
      title: 'Delta Calls',
      dataIndex: 'deltaCalls',
      key: 'deltaCalls',
      sorter: (a: TableRow, b: TableRow) => a.deltaCalls! - b.deltaCalls!
    },
    {
      title: 'Delta Calls%',
      dataIndex: 'deltaCallsPercent',
      key: 'deltaCallsPercent',
      sorter: (a: TableRow, b: TableRow) =>
        a.deltaCallsPercentNumber! - b.deltaCallsPercentNumber!
    }
  ]

  // #endregion

  // #region - State
  const [tableDataSource, setTableDataSource] = React.useState<TableRow[]>([])
  const { run, worker, span, expRun, expWorker, expSpan } = props

  const [columnUnderlyingData, setColumnUnderlyingData] = React.useState<
    ColumnUnderlyingData[]
  >([])

  const [
    rootUnderlyingData,
    setRootUnderlyingData
  ] = React.useState<ColumnUnderlyingData>()

  const [columnChartData, setColumnChartData] = React.useState<any[]>([])
  const [stepChartData, setStepChartData] = React.useState<any[]>([])

  const [
    selectedTableColumnsOptions,
    setSelectedTableColumnsOptions
  ] = React.useState<[key: string]>(['hostDuration'])
  const [selectedTableColumns, setSelectedTableColumns] = React.useState<any[]>(
    [...baseTableColumns, ...hostDurationColumns]
  )

  const [dataStackLevel, setDataStackLevel] = React.useState(0)
  const [loading, setLoading] = React.useState(false)

  // #endregion
  const classes = useStyles()

  // #region - Event Handler
  const handleChartColumnSelect = (row: number, column: number) => {
    if (columnUnderlyingData.length === 0) {
      return
    }

    let selectedUnderlyingData = columnUnderlyingData[row]
    if (!selectedUnderlyingData) {
      return
    }

    let tableDataSource = generateDataSourceFromUnderlyingData(
      selectedUnderlyingData
    )
    setTableDataSource(tableDataSource)
    columnTableDataSourceStack.push(tableDataSource)

    setLoading(true)

    api.defaultApi
      .diffnodeGet(
        run,
        worker,
        span,
        expRun,
        expWorker,
        expSpan,
        selectedUnderlyingData.path
      )
      .then((resp) => handleDiffNodeResp(resp))
      .finally(() => setLoading(false))
  }

  const handleGoBack = () => {
    if (columnChartDataStack.length > 1) {
      columnChartDataStack.pop()
      let top = columnChartDataStack[columnChartDataStack.length - 1]
      setColumnChartData(top)
    }

    if (stepChartDataStack.length > 1) {
      stepChartDataStack.pop()
      let top = stepChartDataStack[stepChartDataStack.length - 1]
      setStepChartData(top)
    }

    if (columnUnderlyingDataStack.length > 0) {
      columnUnderlyingDataStack.pop()
      let top = columnUnderlyingDataStack[columnUnderlyingDataStack.length - 1]
      setColumnUnderlyingData(top)
    }

    if (columnTableDataSourceStack.length > 0) {
      columnTableDataSourceStack.pop()
      let top =
        columnTableDataSourceStack[columnTableDataSourceStack.length - 1]

      if (top) {
        setTableDataSource(top)
      } else {
        let tableDataSource = generateDataSourceFromUnderlyingData(
          rootUnderlyingData!
        )
        setTableDataSource(tableDataSource)
      }
    }

    setDataStackLevel(dataStackLevel - 1)
  }

  const toPercentString = (percentNumber: number) => {
    if (isNaN(percentNumber)) {
      return 'N/A'
    }

    return `${percentNumber.toFixed(2)}%`
  }

  const handleColumnSelectionChange = (value: [key: string]) => {
    let columns = value.map((x) => tableSourceColumnMap[x]).flat()
    let r = [...baseTableColumns, ...columns]
    setSelectedTableColumnsOptions(value)
    setSelectedTableColumns(r)
  }

  const generateDataSourceFromUnderlyingData = (
    selectedUnderlyingData: ColumnUnderlyingData
  ) => {
    let tableDataSource: TableRow[] = []

    for (let i = 0; i < selectedUnderlyingData.leftAggs.length; i++) {
      let left = selectedUnderlyingData.leftAggs[i]
      let right = selectedUnderlyingData.rightAggs[i]

      let deltaCallsPercentNumber =
        ((right.calls - left.calls) / left.calls) * 100

      let deltaHostDurationPercentNumber =
        ((right.host_duration - left.host_duration) / left.host_duration) * 100

      let deltaSelfHostDurationPercentNumber =
        ((right.self_host_duration - left.self_host_duration) /
          left.self_host_duration) *
        100

      let deltaDeviceDurationPercentNumber =
        ((right.device_duration - left.device_duration) /
          left.device_duration) *
        100

      let deltaSelfDeviceDurationPercentNumber =
        ((right.self_device_duration - left.self_device_duration) /
          left.self_device_duration) *
        100

      tableDataSource.push({
        key: i,
        operator: left.name,
        baselineCalls: left.calls,
        expCalls: right.calls,
        deltaCalls: right.calls - left.calls,
        deltaCallsPercentNumber: deltaCallsPercentNumber,
        deltaCallsPercent: toPercentString(deltaCallsPercentNumber),

        baselineHostDuration: left.host_duration,
        expHostDuration: right.host_duration,
        deltaHostDuration: parseFloat((right.host_duration - left.host_duration).toFixed(3)),
        deltaHostDurationPercentNumber: deltaHostDurationPercentNumber,
        deltaHostDurationPercent: toPercentString(
          deltaHostDurationPercentNumber
        ),

        baselineSelfHostDuration: left.self_host_duration,
        expSelfHostDuration: right.self_host_duration,
        deltaSelfHostDuration:
          parseFloat((right.self_host_duration - left.self_host_duration).toFixed(3)),
        deltaSelfHostDurationPercentNumber: deltaSelfHostDurationPercentNumber,
        deltaSelfHostDurationPercent: toPercentString(
          deltaSelfHostDurationPercentNumber
        ),

        baselineDeviceDuration: left.device_duration,
        expDeviceDuration: right.device_duration,
        deltaDeviceDuration: parseFloat((right.device_duration - left.device_duration).toFixed(3)),
        deltaDeviceDurationPercentNumber: deltaDeviceDurationPercentNumber,
        deltaDeviceDurationPercent: toPercentString(
          deltaDeviceDurationPercentNumber
        ),

        baselineSelfDeviceDuration: left.self_device_duration,
        expSelfDeviceDuration: right.self_device_duration,
        deltaSelfDeviceDuration:
          parseFloat((right.self_device_duration - left.self_device_duration).toFixed(3)),
        deltaSelfDeviceDurationPercentNumber: deltaSelfDeviceDurationPercentNumber,
        deltaSelfDeviceDurationPercent: toPercentString(
          deltaSelfDeviceDurationPercentNumber
        )
      })
    }

    return tableDataSource
  }

  React.useEffect(() => {
    if (
      run.length > 0 &&
      worker.length > 0 &&
      span.length > 0 &&
      expRun.length > 0 &&
      expWorker.length > 0 &&
      expSpan.length > 0
    ) {
      setLoading(true)

      columnChartDataStack = []
      stepChartDataStack = []
      columnUnderlyingDataStack = []
      columnTableDataSourceStack = []

      api.defaultApi
        .diffnodeGet(run, worker, span, expRun, expWorker, expSpan)
        .then((resp) => {
          handleDiffNodeResp(resp)
          let rootUnderlyingData = {
            name: 'rootNode',
            path: resp.path,
            leftAggs: resp.left.aggs,
            rightAggs: resp.right.aggs
          }

          setRootUnderlyingData(rootUnderlyingData)
          let tableDataSource = generateDataSourceFromUnderlyingData(
            rootUnderlyingData!
          )
          setTableDataSource(tableDataSource)
        })
        .finally(() => setLoading(false))

      setSelectedTableColumns([...baseTableColumns, ...hostDurationColumns])
    }
  }, [run, worker, span, expRun, expWorker, expSpan])

  const handleDiffNodeResp = (resp: any) => {
    let columnChartData: any[] = []
    let stepChartData: any[] = []
    let underlyingData: ColumnUnderlyingData[] = []

    columnChartData.push([
      'Call',
      'Baseline',
      'Experiment',
      'Baseline Trend',
      'Exp Trend'
    ])
    stepChartData.push(['Call', 'Diff', 'Accumulated Diff'])

    if (resp.children.length > 0) {
      let accumulated_left_duration = 0
      let accumulated_right_duration = 0
      let accumulated_step_diff = 0
      for (let i = 0; i < resp.children.length; i++) {
        let left = resp.children[i].left
        let right = resp.children[i].right
        let currColumn: any[] = []
        let currStep: any[] = []

        let name = left.name
        if (name === COMPOSITE_NODES_NAME) {
          continue
        }

        if (name.startsWith('aten::')) {
          // Ignore aten operators
          continue
        }

        if (name.startsWith('enumerate(DataLoader)')) {
          name = name.substring(21)
        }

        if (name.startsWith('enumerate(DataPipe)')) {
          name = name.substring(19)
        }

        if (name.startsWith('nn.Module: ')) {
          name = name.substring(11)
        }

        if (name.startsWith('Optimizer.zero_grad')) {
          name = 'Optimizer.zero_grad'
        }

        if (name.startsWith('Optimizer.step')) {
          name = 'Optimizer.step'
        }

        currColumn.push(name)
        currColumn.push(left.total_duration)
        currColumn.push(right.total_duration)

        accumulated_left_duration += left.total_duration
        currColumn.push(accumulated_left_duration)

        accumulated_right_duration += right.total_duration
        currColumn.push(accumulated_right_duration)
        columnChartData.push(currColumn)

        underlyingData.push({
          name: name,
          path: resp.children[i].path,
          leftAggs: left.aggs,
          rightAggs: right.aggs
        })

        currStep.push(name)
        let stepDiff = right.total_duration - left.total_duration
        currStep.push(stepDiff)

        accumulated_step_diff += stepDiff
        currStep.push(accumulated_step_diff)

        stepChartData.push(currStep)
      }
    } else {
      let left = resp.left
      let right = resp.right
      let currColumn: any[] = []
      let currStep: any[] = []
      let name = left.name

      if (name.startsWith('nn.Module: ')) {
        name = name.substring(11)
      }

      currColumn.push(name)
      currColumn.push(left.total_duration)
      currColumn.push(right.total_duration)
      currColumn.push(left.total_duration)
      currColumn.push(right.total_duration)

      columnChartData.push(currColumn)

      currStep.push(name)
      let stepDiff = right.total_duration - left.total_duration
      currStep.push(stepDiff)
      currStep.push(stepDiff)
      stepChartData.push(currStep)
    }

    setColumnChartData(columnChartData)
    columnChartDataStack.push(columnChartData)

    setStepChartData(stepChartData)
    stepChartDataStack.push(stepChartData)

    setColumnUnderlyingData(underlyingData)
    columnUnderlyingDataStack.push(underlyingData)

    setDataStackLevel(columnChartDataStack.length)
  }

  // #endregion

  if (!loading && columnUnderlyingDataStack.length === 0) {
    return (
      <Card variant="outlined">
        <CardHeader title="No Runs Found"></CardHeader>
        <CardContent>
          <Typography>There is no run selected for diff.</Typography>
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return <FullCircularProgress />
  }

  return (
    <div className={classes.root}>
      <Grid container spacing={1}>
        <Grid container item spacing={1}>
          <Grid item sm={12}>
            <Card variant="outlined">
              <CardHeader title="DiffView" />
              <CardContent>
                <Button
                  className={classes.iconButton}
                  startIcon={<ChevronLeftIcon />}
                  onClick={handleGoBack}
                  variant="outlined"
                  disabled={dataStackLevel < 2}
                >
                  Go Back
                </Button>
                {columnChartData.length > 1 && (
                  <>
                    <DiffColumnChart
                      rawData={columnChartData}
                      selectCallback={handleChartColumnSelect}
                    />
                    <DiffStepChart rawData={stepChartData} />
                  </>
                )}
                {columnChartData.length === 1 && (
                  <Typography>No more level to show.</Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        <Grid container item spacing={1}>
          <Grid item sm={12}>
            <Card variant="outlined">
              <CardHeader title="Operator View" />
              <CardContent>
                <Select
                  mode="multiple"
                  style={{ width: '100%' }}
                  placeholder="Select the data you need"
                  value={selectedTableColumnsOptions}
                  onChange={handleColumnSelectionChange}
                  optionLabelProp="label"
                >
                  <Option value="hostDuration" label="Host Duration">
                    <div>Host Duration</div>
                  </Option>
                  <Option value="selfHostDuration" label="Self Host Duration">
                    <div>Self Host Duration</div>
                  </Option>
                  <Option value="deviceDuration" label="Device Duration">
                    <div>Device Duration</div>
                  </Option>
                  <Option
                    value="selfDeviceDuration"
                    label="Self Device Duration"
                  >
                    <div>Self Device Duration</div>
                  </Option>
                </Select>
                &nbsp;
                <Table
                  dataSource={tableDataSource}
                  columns={selectedTableColumns}
                />
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Grid>
    </div>
  )
}
