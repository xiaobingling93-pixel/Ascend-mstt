/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/**
 * @fileoverview Interfaces that parallel proto definitions in
 * third_party/tensorflow/core/framework/...
 *     graph.proto
 *     step_stats.proto
 * These should stay in sync.
 *
 * When adding a repeated field to this file, make sure to update the
 * GRAPH_REPEATED_FIELDS and METADATA_REPEATED_FIELDS lists within parser.ts.
 * Otherwise, the parser has no way of differentiating between a field with a
 * certain value and a repeated field that has only 1 occurence, resulting in
 * subtle bugs.
 */

export enum NodeOpType {
  MODULE = 0,
  DEFAULT = 1,
  MULTI_COLLECTION = 8,
  API_LIST = 9,
}
/** Name of the node */
export interface NodeDef {
  name: string;
  /** List of nodes that are inputs for this node. */
  input: string[];
  /** The name of the operation associated with this node. */
  op: string;
  /** The op type of the node. */
  node_type: NodeOpType;
  /** The array of inputs data in JSON string format. */
  input_data: {
    [key: string]: any;
  };
  /** The array of outputs data in JSON string format. */
  output_data: {
    [key: string]: any;
  };
  stack_info: [];
  matched_node_link: [];
  suggestions: {
    [key: string]: string;
  };
  /** The array consist of the path of linked node in graph comparison. */
  subnodes?: string[];
  isLeaf: boolean;
  /** List of attributes that describe/modify the operation. */
  attr: {
    key: string;
    value: Record<string, unknown>;
  }[];
}
export interface EdgeInfo {
  input: string;
  output: string;
  shape: string;
  attr: {
    key: string;
    value: Record<string, unknown>;
  }[];
}
/**
 * TensorFlow graph definition as defined in the graph.proto file.
 */
export interface GraphDef {
  // A list of nodes in the graph.
  node: NodeDef[];
  // The information of the list of edges.
  edge?: EdgeInfo[];
}
