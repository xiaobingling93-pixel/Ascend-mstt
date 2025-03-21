/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
import { NPU_PREFIX, BENCH_PREFIX, type ProgressTracker } from './common';
import * as tf_graph from './graph';
import * as hierarchy from './hierarchy';
import * as parser from './parser';
import { GraphDef } from './proto';
import * as tf_graph_util from './util';

export interface GraphAndHierarchy {
  graph: tf_graph.MergedSlimGraph;
  graphHierarchy: hierarchy.MergedHierarchy;
}
export function fetchAndConstructHierarchicalGraph(
  tracker: ProgressTracker,
  remotePath: string | null,
  pbTxtFile: Blob | null,
  hierarchyParams: hierarchy.HierarchyParams = hierarchy.DefaultHierarchyParams,
): Promise<GraphAndHierarchy> {
  const dataTracker = tf_graph_util.getSubtaskTracker(tracker, 30, 'Data');
  const graphTracker = tf_graph_util.getSubtaskTracker(tracker, 20, 'Graph');
  const hierarchyTracker = tf_graph_util.getSubtaskTracker(tracker, 50, 'Namespace hierarchy');
  return parser
    .fetchAndParseGraphData(remotePath as string, pbTxtFile, dataTracker)
    .then(
      (graph): any => {
        if (graph.node.length !== 2) {
          return tf_graph.build(graph, tf_graph.DefaultBuildParams, graphTracker);
        }
        const npuGraph: GraphDef = { node: [] };
        const benchGraph: GraphDef = { node: [] };
        if (graph.node[0].name.startsWith(NPU_PREFIX) && graph.node[1].name.startsWith(BENCH_PREFIX)) {
          npuGraph.node.push(graph.node[0]);
          benchGraph.node.push(graph.node[1]);
        }
        if (graph.node[0].name.startsWith(BENCH_PREFIX) && graph.node[1].name.startsWith(NPU_PREFIX)) {
          npuGraph.node.push(graph.node[1]);
          benchGraph.node.push(graph.node[0]);
        }
        return Promise.all([
          tf_graph.build(npuGraph, tf_graph.DefaultBuildParams, graphTracker),
          tf_graph.build(benchGraph, tf_graph.DefaultBuildParams, graphTracker),
        ]);
      },
      () => {
        throw new Error(
          'Malformed GraphDef. This can sometimes be caused by ' +
            'a bad network connection or invalid inputting files ',
        );
      },
    )
    .then(async (graph) => {
      if (Array.isArray(graph)) {
        const mergedGraph: tf_graph.MergedSlimGraph = { npu: graph[0], bench: graph[1] };
        const npuHierarchy = await hierarchy.build(graph[0], hierarchyParams, hierarchyTracker);
        const benchHierarchy = await hierarchy.build(graph[1], hierarchyParams, hierarchyTracker);
        return { graph: mergedGraph, graphHierarchy: { npu: npuHierarchy, bench: benchHierarchy } };
      }
      const graphHierarchy = await hierarchy.build(graph, hierarchyParams, hierarchyTracker);
      return { graph: { npu: graph }, graphHierarchy: { npu: graphHierarchy } };
    })
    .catch((e) => {
      // Generic error catch, for errors that happened outside
      // asynchronous tasks.
      const msg = `Graph visualization failed.\n\n${e}`;
      tracker.reportError(msg, e);
      // Don't swallow the error.
      throw e;
    });
}
