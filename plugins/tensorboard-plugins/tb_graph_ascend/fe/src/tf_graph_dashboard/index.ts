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

Copyright (c) 2025, Huawei Technologies.
Adapt to the model hierarchical visualization data collected by the msprobe tool
==============================================================================*/

import { customElement, observe, property } from '@polymer/decorators';
import { html, PolymerElement } from '@polymer/polymer';
import '../polymer/irons_and_papers';
import { LegacyElementMixin } from '../polymer/legacy_element_mixin';
import { Canceller } from '../tf_backend/canceller';
import { RequestManager } from '../tf_backend/requestManager';
import '../tf_dashboard_common/tf-dashboard-layout';
import * as tf_storage from '../tf_storage';
import * as vz_sorting from '../vz_sorting/sorting';
import '../tf_graph_board/tf-graph-board';
import * as tf_graph_render from '../tf_graph_common/render';
import '../tf_graph_controls/tf-graph-controls';
import '../tf_graph_loader/tf-graph-dashboard-loader';

/**
 * The (string) name for the run of the selected dataset in the graph dashboard.
 */
const RUN_STORAGE_KEY = 'run';
/**
 * TODO(stephanwlee): Convert this to proper type when converting to TypeScript.
 * @typedef {{
 *   tag: ?string,
 *   displayName: string,
 *   conceptualGraph: boolean,
 *   opGraph: boolean,
 *   profile: boolean,
 * }}
 */
const TagItem = {};
/**
 * TODO(stephanwlee): Convert this to proper type when converting to TypeScript.
 * @typedef {{
 *   name: string,
 *   tags: !Array<!TagItem>,
 * }}
 */
const RunItem = {};

/**
 * tf-graph-dashboard displays a graph from a TensorFlow run.
 *
 * It has simple behavior: Creates a url-generator and run-generator
 * to talk to the backend, and then passes the runsWithGraph (list of runs with
 * associated graphs) along with the url generator into tf-graph-board for display.
 *
 * If there are multiple runs with graphs, the first run's graph is shown
 * by default. The user can select a different run from a dropdown menu.
 */
@customElement('graph-app')
class TfGraphDashboard extends LegacyElementMixin(PolymerElement) {
  static readonly template = html`
    <paper-dialog id="error-dialog" with-backdrop></paper-dialog>
    <tf-dashboard-layout>
      <tf-graph-controls
        id="controls"
        class="sidebar"
        slot="sidebar"
        menu="[[_menu]]"
        colors="[[_colors]]"
        colorset="[[_colorset]]"
        overflowcheck="[[_overflowcheck]]"
        microsteps="[[_microsteps]]"
        steplist="[[_steplist]]"
        unmatched="[[_unmatched]]"
        matchedlist="[[_matchedlist]]"
        datasets="[[_datasets]]"
        render-hierarchy="[[_renderHierarchy]]"
        selection="{{_selection}}"
        selected-file="{{_selectedFile}}"
        selected-node="{{_selectedNode}}"
        on-fit-tap="_fit"
        trace-inputs="{{_traceInputs}}"
        auto-extract-nodes="{{_autoExtractNodes}}"
        on-download-image-requested="_onDownloadImageRequested"
      ></tf-graph-controls>
      <div class$="center [[_getGraphDisplayClassName(_selectedFile, _datasets)]]" slot="center">
        <tf-graph-dashboard-loader
          id="loader"
          datasets="[[_datasets]]"
          selection="[[_selection]]"
          selected-file="[[_selectedFile]]"
          out-graph-hierarchy="{{_graphHierarchy}}"
          out-graph="{{_graph}}"
          out-stats="{{_stats}}"
          tooltips="{{_tooltips}}"
          menu="{{_menu}}"
          colorset="{{_colorset}}"
          colors="{{_colors}}"
          overflowcheck="{{_overflowcheck}}"
          microsteps="{{_microsteps}}"
          steplist="{{_steplist}}"
          unmatched="{{_unmatched}}"
          matchedlist="{{_matchedlist}}"
          progress="{{_progress}}"
          hierarchy-params="[[_hierarchyParams]]"
        ></tf-graph-dashboard-loader>
        <div class="no-data-message">
          <h3>No graph definition files were found.</h3>
        </div>
        <div class="graphboard">
          <tf-graph-board
            id="graphboard"
            selection="[[_selection]]"
            graph-hierarchy="[[_graphHierarchy]]"
            graph="[[_graph]]"
            hierarchy-params="[[_hierarchyParams]]"
            progress="[[_progress]]"
            debugger-data-enabled="[[_debuggerDataEnabled]]"
            debugger-numeric-alerts="[[_debuggerNumericAlerts]]"
            all-steps-mode-enabled="{{allStepsModeEnabled}}"
            render-hierarchy="{{_renderHierarchy}}"
            selected-node="{{_selectedNode}}"
            stats="[[_stats]]"
            trace-inputs="[[_traceInputs]]"
            auto-extract-nodes="[[_autoExtractNodes]]"
            tooltips="[[_tooltips]]"
          ></tf-graph-board>
        </div>
      </div>
    </tf-dashboard-layout>
    <style>
      :host /deep/ {
        font-family: 'Roboto', sans-serif;
      }

      .sidebar {
        display: flex;
        height: 100%;
      }

      .center {
        position: relative;
        height: 100%;
      }

      paper-dialog {
        padding: 20px;
      }

      .no-data-message {
        max-width: 540px;
        margin: 80px auto 0 auto;
      }

      .graphboard {
        height: 100%;
      }

      .no-graph .graphboard {
        display: none;
      }

      .center:not(.no-graph) .no-data-message {
        display: none;
      }

      a {
        color: var(--tb-link);
      }

      a:visited {
        color: var(--tb-link-visited);
      }
    </style>
  `;

  /**
   * @type {!Array<!RunItem>}
   */
  @property({ type: Array })
  _datasets: any[] = [];

  @property({ type: Boolean })
  _datasetsFetched: boolean = false;

  @property({ type: Number })
  _selectedDataset: number = 0;

  @property({ type: Object })
  _renderHierarchy: tf_graph_render.RenderGraphInfo;

  @property({ type: Object })
  _requestManager: RequestManager = new RequestManager();

  @property({ type: Object })
  _canceller: Canceller = new Canceller();

  @property({ type: Boolean })
  _debuggerDataEnabled: boolean;

  @property({ type: Boolean })
  allStepsModeEnabled: boolean;

  @property({ type: String, notify: true })
  selectedNode: string;

  @property({ type: Boolean })
  _isAttached: boolean;

  // Whether this dashboard is initialized. This dashboard should only be initialized once.
  @property({ type: Boolean })
  _initialized: boolean;

  // An array of alerts (in chronological order) provided by debugging libraries on when bad
  // values (NaN, +/- Inf) appear.
  @property({ type: Array, notify: true })
  _debuggerNumericAlerts: unknown[] = [];

  @property({ type: Array })
  runs: unknown[];

  @property({
    type: String,
    notify: true,
    observer: '_runObserver',
  })
  run: string = tf_storage
    .getStringInitializer(RUN_STORAGE_KEY, {
      defaultValue: '',
      useLocalStorage: false,
    })
    .call(this);

  @property({ type: Object })
  _selection: object;

  @property({ type: Boolean })
  _traceInputs: boolean;

  @property({ type: Boolean })
  _autoExtractNodes: boolean;

  @property({ type: Object })
  _selectedFile: any;

  _runObserver = tf_storage.getStringObserver(RUN_STORAGE_KEY, {
    defaultValue: '',
    polymerProperty: 'run',
    useLocalStorage: false,
  });

  @observe('_isAttached')
  _maybeInitializeDashboard(): void {
    let isAttached = this._isAttached;
    if (this._initialized || !isAttached) {
      // Either this dashboard is already initialized ... or we are not yet ready to initialize.
      return;
    }
    // Set this to true so we only initialize once.
    this._initialized = true;
    this._fetchDataset().then((dataset) => {
      const runNames = Object.keys(dataset);
      // Transform raw data into UI friendly data.
      this._datasets = runNames.sort(vz_sorting.compareTagNames).map((runName) => {
        const runData = dataset[runName];
        const tagNames = Object.keys(runData.tags).sort(vz_sorting.compareTagNames);
        const tags = tagNames
          .map((name) => runData.tags[name])
          .map(({ tag, conceptual_graph, op_graph, profile }) => ({
            tag,
            displayName: tag,
            conceptualGraph: conceptual_graph,
            opGraph: op_graph,
            profile,
          }));
        return { name: runName, tags };
      });
      this._datasetsFetched = true;
    });
  }

  @observe('_datasetsFetched', '_datasets', 'run')
  _determineSelectedDataset(): void {
    let datasetsFetched = this._datasetsFetched;
    let datasets = this._datasets;
    let run = this.run;
    // By default, load the first dataset.
    if (!run) {
      // By default, load the first dataset.
      this.set('_selectedDataset', 0);
      return;
    }
    // If the URL specifies a dataset, load it.
    const dataset = datasets.findIndex((d) => d.name === run);
    if (dataset === -1) {
      if (datasetsFetched) {
        // Tell the user if the dataset cannot be found to avoid misleading
        // the user.
        const dialog = this.$$('#error-dialog') as any;
        dialog.textContent = `No dataset named "${run}" could be found.`;
        dialog.open();
      }
      return;
    }
    this.set('_selectedDataset', dataset);
  }

  @observe('_datasetsFetched', '_datasets', '_selectedDataset')
  _updateSelectedDatasetName(): void {
    let datasetsFetched = this._datasetsFetched;
    let datasets = this._datasets;
    let selectedDataset = this._selectedDataset;
    if (!datasetsFetched) {
      return;
    }
    // Cannot update `run` to update the hash in case datasets for graph is empty.
    if (datasets.length <= selectedDataset) {
      return;
    }
    this.set('run', datasets[selectedDataset].name);
  }

  override attached(): void {
    this.set('_isAttached', true);
  }

  override detached(): void {
    this.set('_isAttached', false);
  }

  ready(): void {
    super.ready();
  }

  _fit(): void {
    (this.$$('#graphboard') as any).fit();
  }

  _onDownloadImageRequested(event: CustomEvent): void {
    (this.$$('#graphboard') as any).downloadAsImage(event.detail as string);
  }

  _getGraphDisplayClassName(_selectedFile: any, _datasets: any[]): string {
    const isDataValid = _selectedFile || _datasets.length;
    return isDataValid ? '' : 'no-graph';
  }

  _fetchDataset(): Promise<any> {
    return this._requestManager.request('info');
  }

  _datasetsState(datasetsFetched, datasets, state): boolean {
    if (!datasetsFetched) {
      return state === 'NOT_LOADED';
    }
    if (!datasets || !datasets.length) {
      return state === 'EMPTY';
    }
    return state === 'PRESENT';
  }
}
