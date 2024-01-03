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
 * Modifications: Offer offline supporting.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Graph } from '../../api'
import { value } from '../../utils'
import { useResizeEventDependency } from '../../utils/resize'
import * as echarts from 'echarts'

interface IProps {
  graph: Graph
  height?: number
  top?: number
  noLegend?: boolean
  title?: string
  colors?: Array<string>
  tooltip_mode?: string
}

const noLegendArea = { left: '5%', width: '90%', top: '5%', height: '90%' }
const normalArea = { left: '5%', width: '95%' }
const noTitleArea = { left: '5%', width: '95%', top: '10%', height: '80%' }

export const PieChart: React.FC<IProps> = (props) => {
  const {
    graph,
    height = 300,
    top,
    noLegend,
    title,
    colors,
    tooltip_mode = 'both'
  } = props
  const graphRef = React.useRef<HTMLDivElement>(null)

  const [resizeEventDependency] = useResizeEventDependency()

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    const chart = echarts.init(element)

    let totalValue = 0
    const rowsWithUniqueName: Array<{ name: string, value: number }> =
      top === undefined
        ? graph.rows.map((item, index) => {
          totalValue += item[1] as number
          return { name: `${index}_${item[0]}`, value: item[1] as number }
        })
        : graph.rows
          .sort((a, b) => (value(b[1]) as number) - (value(a[1]) as number))
          .slice(0, top).map((item, index) => {
            totalValue += item[1] as number
            return { name: `${index}_${item[0]}`, value: item[1] as number }
          })

    const option: echarts.EChartsOption = {
      height,
      width: '100%',
      title: {
        text: title
      },
      tooltip: {
        trigger: 'item',
        formatter: (data) => {
          const typedData = data as echarts.DefaultLabelFormatterCallbackParams
          const index = typedData.name.indexOf('_')
          const safeName = typedData.name.replace(/</g, '&lt;').replace(/>/g, '&gt;')
          return `${index > -1 ? safeName.slice(index + 1) : safeName}<br /><b>${tooltip_mode === 'both' ?
              typedData.value : ''}(${typedData.percent}%)<b />`
        },
        confine: true,
        extraCssText: `max-width: 300px;
          word-wrap:break-word;
          white-space:pre-wrap;
          padding-right: 10px`
      },
      chartArea: noLegend ? noLegendArea : !title ? noTitleArea : normalArea,
      legend: {
        type: noLegend ? 'plain' : 'scroll',
        orient: 'vertical',
        left: 'right',
        z: 10,
        // Display at most 36 characters.
        formatter: (name) => {
          // Show legends for datas with the same name.
          const index = name.indexOf('_')
          if (index > -1) {
            name = name.slice(index + 1)
          }
          return name.length > 36 ? name.slice(0, 34) + "..." : name;
        },
        tooltip: {
          show: true,
          triggerOn: 'mousemove',
          formatter: (data) => {
            const currentItem = rowsWithUniqueName.find(item => item.name === data.name)
            const index = data.name.indexOf('_')
            const percent = ((currentItem?.value || 0) * 100 / totalValue).toFixed(2)
            const safeName = data.name.replace(/</g, '&lt;').replace(/>/g, '&gt;')
            return `${index > -1 ? safeName.slice(index + 1) :
              safeName}<br /><b>${tooltip_mode === 'both' ? (currentItem?.value || 0) : ''}(${percent}%)<b />`
          }
        }
      },
      sliceVisibilityThreshold: 0,
      colors,
      series: [
        {
          type: 'pie',
          radius: ['32%', '80%'],
          center: ['32%', '50%'],
          label: {
            position: 'inside',
            formatter: `{d}%`,
            color: '#ffffff'
          },
          data: rowsWithUniqueName
        }
      ]
    }

    option && chart.setOption(option, true)

    return () => {
      chart.dispose()
    }
  }, [graph, height, top, resizeEventDependency])

  return (
    <div ref={graphRef} style={{ height }}></div>
  )
}
