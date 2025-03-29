/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import { customElement, observe, property } from '@polymer/decorators';
import { PolymerElement } from '@polymer/polymer';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import * as tf_graph_common from '../tf_graph_common/common';
import * as tf_graph from '../tf_graph_common/graph';
import * as tf_graph_hierarchy from '../tf_graph_common/hierarchy';
import * as tf_graph_loader from '../tf_graph_common/loader';
import * as tf_graph_parser from '../tf_graph_common/parser';
import * as tf_graph_util from '../tf_graph_common/util';
import * as tf_graph_node from '../tf_graph_common/node';
import { DATA_LOAD_TIME, DATA_NOTICE_TIME } from '../tf_graph_common/common';
import * as tf_graph_controls from '../tf_graph_controls/tf-graph-controls';
import { safeJSONParse } from '../utils';

interface GraphRunTag {
  run: string;
  tag: string | null;
}

interface Components {
  menu: object;
  tooltips: string;
  colors: object;
  overflowCheck: boolean;
  microSteps: number;
  stepList: [];
  unMatchedNode: [];
  match: [];
}
/**
 * Data loader for tf-graph-dashboard.
 *
 * The loader loads op graph, conceptual graphs, and RunMetadata associated with
 * an op graph which is the major difference from the tf-graph-loader which is
 * only capable of loading an op graph. Another difference is that the loader
 * takes `selection` from the tf-graph-controls as an input as opposed to URL
 * path of an data endpoint.
 */
@customElement('tf-graph-dashboard-loader')
class TfGraphDashboardLoader extends LegacyElementMixin(PolymerElement) {
  static readonly _template = null;

  @property({ type: Array })
  datasets: any[];

  /**
   * @type {{value: number, msg: string}}
   *
   * A number between 0 and 100 denoting the % of progress
   * for the progress bar and the displayed message.
   */
  @property({ type: Object, notify: true })
  progress: object;

  @property({ type: Object })
  selection: any;

  /**
   * @type {?Event}
   */
  @property({ type: Object })
  selectedFile: object;

  @property({ type: Object })
  hierarchyParams = tf_graph_hierarchy.defaultHierarchyParams;

  @property({
    type: Object,
    readOnly: true, // readonly so outsider can't change this via binding
    notify: true,
  })
  outGraphHierarchy: tf_graph_hierarchy.Hierarchy;

  @property({
    type: Object,
    readOnly: true, // readonly so outsider can't change this via binding
    notify: true,
  })
  outGraph: tf_graph.SlimGraph;

  @property({
    type: Object,
    readOnly: true, // This property produces data.
    notify: true,
  })
  outStats: object;

  @property({ type: Object })
  _graphRunTag: GraphRunTag;

  @property({
    type: Object,
    notify: true,
  })
  menu: object;

  @property({
    type: Object,
    notify: true,
  })
  colorset: object;

  @property({
    type: Object,
    notify: true,
  })
  tooltips: object;

  @property({
    type: Object,
    notify: true,
  })
  colors: any;

  @property({
    type: Array,
    notify: true,
  })
  overflowcheck;

  @property({
    type: Object,
    notify: true,
  })
  microsteps: any;

  @property({
    type: Object,
    notify: true,
  })
  steplist: any;

  @property({
    type: Object,
    notify: true,
  })
  unmatched: object;

  @property({
    type: Object,
    notify: true,
  })
  matchedlist: object;

  @observe('selectedFile')
  _selectedFileChanged(): void {
    let e = this.selectedFile;
    if (!e) {
      return;
    }
    const target = (e as any).target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) {
      return;
    }
    // Clear out the value of the file chooser. This ensures that if the user
    // selects the same file, we'll re-read it.
    target.value = '';
    this._fetchAndConstructHierarchicalGraph(null, file);
  }

  @observe('selection')
  _selectionChanged(): void {
    if (!this.selection) {
      return;
    }
    // selection can change a lot within a microtask.
    // Don't fetch too much too fast and introduce race condition.
    this.debounce('selectionchange', () => {
      this._load(this.selection);
    });
  }

  getColors(): any {
    return this.colors;
  }
  _setCompoments(componentsPath): Promise<void> {
    return new Promise<void>(async (resolve, reject) => {
      this.set('progress', {
        value: 0,
        msg: '',
      });

      const tracker = tf_graph_util.getTracker(this);
      const dataTracker = tf_graph_util.getSubtaskTracker(tracker, 100, 'Data');
      dataTracker.setMessage('Initialization in progress');

      let timer = 0;
      let shouldBreak = false; // 标志位，用于控制循环退出

      // 启动定时器任务
      const timerTask = async function (): Promise<void> {
        let previousProgress = 0; // 记录上一次更新的进度

        while (timer <= DATA_LOAD_TIME && !shouldBreak) {
          if (timer < DATA_NOTICE_TIME) {
            const progress = Math.log(timer + 1) / Math.log(DATA_NOTICE_TIME);
            const progressIncrement = (progress * 100) - previousProgress;
            dataTracker.updateProgress(progressIncrement);
            previousProgress = progress * 100;
          } else {
            dataTracker.setMessage('File data too large, still reading');
          }
          await new Promise((resolveTimer) => setTimeout(resolveTimer, 100));
          timer++;
        }
      }.bind(this);

      const fetchTask = async (): Promise<void> => {
        let componentsStr;
        try {
          componentsStr = await tf_graph_parser.fetchPbTxt(componentsPath);
        } catch (e) {
          shouldBreak = true; // 捕获 fetchPbTxt 错误并停止定时器
          dataTracker.reportError('Fetch error, please check first file in file path', e as Error);
          return;
        }

        shouldBreak = true; // 正常流程也停止定时器

        let components: Components = {
          menu: [],
          tooltips: '',
          colors: {},
          overflowCheck: false,
          microSteps: 0,
          stepList: [],
          unMatchedNode: [],
          match: [],
        };

        try {
          if (componentsStr) {
            components = {
              ...components,
              ...(safeJSONParse(new TextDecoder().decode(componentsStr).replace(/'/g, '"')) as Components),
            };
          }
        } catch (e) {
          shouldBreak = true; // 解析错误时停止定时器
          dataTracker.reportError(
            'Parse components failed, please check the format of config data in the input vis file',
            e as Error,
          );
          return;
        }
        // 后续处理逻辑...
        const entries = Object.entries(components.tooltips || {});
        const toolTipObject = Object.fromEntries(entries);

        this.set('menu', components.menu);
        this.set('tooltips', toolTipObject);
        this.set('colors', components.colors);
        this.set('overflowcheck', components.overflowCheck);
        this.set('colorset', Object.entries(components.colors || {}));
        this.set('unmatched', components.unMatchedNode);
        this.set('matchedlist', components.match);

        tf_graph_node.getColors(components.colors);

        const microstepsCount = Number(components.microSteps);
        if (microstepsCount) {
          const microstepsArray = ['ALL', ...Array.from({ length: microstepsCount }, (_, index) => index)];
          this.set('microsteps', microstepsArray);
        } else {
          this.set('microsteps', []);
        }
        const steplistCount = Number(components.microSteps);
        this.set('steplist', steplistCount ? components.stepList : []);
        resolve();
      }

      // 同时启动定时器和 fetch 任务
      await Promise.all([timerTask(), fetchTask()]);
    });
  }

  _load(selection: tf_graph_controls.Selection): void {
    const { run, tag, type: selectionType, batch, step } = selection;
    switch (selectionType) {
      case tf_graph_common.SelectionType.OP_GRAPH:
      case tf_graph_common.SelectionType.CONCEPTUAL_GRAPH: {
        // Clear stats about the previous graph.
        this.set('outStats', null);
        const params = new URLSearchParams();
        params.set('run', run);
        params.set('conceptual', String(selectionType === tf_graph_common.SelectionType.CONCEPTUAL_GRAPH));
        if (tag) {
          params.set('tag', tag);
        }
        params.set('batch', String(batch === -1 ? -1 : batch - 1));
        params.set('step', String(step === -1 ? -1 : step - 1));
        const componentsPath = `components?${String(params)}`;
        params.set('node', 'root');
        const graphPath = `subgraph?${String(params)}`;
        this._setCompoments(componentsPath).then(() => {
          // _setCompoments 完成后执行此行
          this._fetchAndConstructHierarchicalGraph(graphPath).then(() => {
            this._graphRunTag = { run, tag }; // 图形构建完成后执行
          });
        });
        return;
      }
      default:
        console.error(`Unknown selection type: ${selectionType}`);
    }
  }

  _fetchAndConstructHierarchicalGraph(path: string | null, pbTxtFile?: Blob): Promise<void> {
    // Reset the progress bar to 0.
    this.set('progress', {
      value: 0,
      msg: '',
    });
    const tracker = tf_graph_util.getTracker(this);
    return tf_graph_loader
      .fetchAndConstructHierarchicalGraph(
        tracker,
        path,
        pbTxtFile !== undefined ? pbTxtFile : null,
        this.hierarchyParams,
      )
      .then(({ graph, graphHierarchy }): void => {
        this._setOutGraph(graph);
        this._setOutGraphHierarchy(graphHierarchy);
      },
      );
  }
}
