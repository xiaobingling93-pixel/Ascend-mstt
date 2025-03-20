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

Copyright (c) 2025, Huawei Technologies.
Adapt to the model hierarchical visualization data collected by the msprobe tool
==============================================================================*/
import * as tb_debug from '../tb_debug';
import { safeJSONParse } from '../utils';
import { ProgressTracker } from './common';
import * as tf_graph_proto from './proto';
import * as tf_graph_util from './util';

function parseValue(value: string): string | number | boolean {
  if (value === 'true') {
    return true;
  }
  if (value === 'false') {
    return false;
  }
  let firstChar = value[0];
  if (firstChar === '"') {
    return value.substring(1, value.length - 1);
  }
  let num = parseFloat(value);
  return isNaN(num) ? value : num;
}
/**
 * Fetches a text file and returns a promise of the result.
 */
export function fetchPbTxt(filepath: string): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    fetch(filepath).then((res) => {
      // Fetch does not reject for 400+.
      if (res.ok) {
        res.arrayBuffer().then(resolve, reject);
      } else {
        res.text().then(reject, reject);
      }
    });
  });
}
/**
 * Fetches the metadata file, parses it and returns a promise of the result.
 */
export function fetchAndParseMetadata(
  path: string,
  tracker: ProgressTracker,
): Promise<tf_graph_proto.StepStats | null> {
  return tf_graph_util
    .runTask(
      'Reading metadata pbtxt',
      40,
      () => {
        if (path == null) {
          return Promise.resolve(null);
        }
        return fetchPbTxt(path);
      },
      tracker,
      tb_debug.GraphDebugEventId.FETCH_METADATA_PBTXT_BYTES,
    )
    .then((arrayBuffer: ArrayBuffer | null) => {
      return tf_graph_util.runAsyncPromiseTask(
        'Parsing metadata.pbtxt',
        60,
        () => {
          return arrayBuffer != null ? parseStatsPbTxt(arrayBuffer) : Promise.resolve(null);
        },
        tracker,
        tb_debug.GraphDebugEventId.PARSE_METADATA_PBTXT_INTO_OBJECT,
      );
    });
}
/**
 * Fetches the graph file, parses it and returns a promise of the result. The
 * result will be undefined if the graph is empty.
 */
export function fetchAndParseGraphData(
  path: string,
  pbTxtFile: Blob | null,
  tracker: ProgressTracker,
): Promise<tf_graph_proto.GraphDef> {
  return tf_graph_util
    .runAsyncPromiseTask(
      'Reading graph pbtxt',
      40,
      async () => {
        if (pbTxtFile) {
          const result = await new Promise<ArrayBuffer>((resolve, reject) => {
            let fileReader = new FileReader();
            fileReader.onload = (): void => resolve(fileReader.result as ArrayBuffer);
            fileReader.onerror = (): void => reject(fileReader.error);
            fileReader.readAsArrayBuffer(pbTxtFile);
          });
          return result;
        }

        const result = await fetchPbTxt(path);
        return result;
      },
      tracker,
      tb_debug.GraphDebugEventId.FETCH_PBTXT_BYTES,
    )
    .then((arrayBuffer: ArrayBuffer) => {
      return tf_graph_util.runAsyncPromiseTask(
        'Parsing graph.pbtxt',
        60,
        () => {
          return parseGraphPbTxt(arrayBuffer);
        },
        tracker,
        tb_debug.GraphDebugEventId.PARSE_PBTXT_INTO_OBJECT,
      );
    });
}
/**
 * Parse a file object in a streaming fashion line by line (or custom delim).
 * Can handle very large files.
 * @param input The file object as an array buffer.
 * @param callback The callback called on each line
 * @param chunkSize The size of each read chunk. (optional)
 * @param delim The delimiter used to split a line. (optional)
 * @returns Promise that resolves with true when it is finished.
 */
export function streamParse(
  arrayBuffer: ArrayBuffer,
  callback: (string) => void,
  chunkSize: number = 1000000,
  delim: string = '\n',
): Promise<boolean> {
  return new Promise<boolean>((resolve, reject) => {
    function readChunk(oldData: string, newData: string, offset: number): void {
      const doneReading = offset >= arrayBuffer.byteLength;
      const parts = newData.split(delim);
      parts[0] = oldData + parts[0];
      // The last part may be part of a longer string that got cut off
      // due to the chunking.
      const remainder = doneReading ? '' : parts.pop();
      for (let part of parts) {
        try {
          callback(part);
        } catch (e) {
          reject(e);
          return;
        }
      }
      if (doneReading) {
        resolve(true);
        return;
      }
      const nextChunk = new Blob([arrayBuffer.slice(offset, offset + chunkSize)]);
      const file = new FileReader();
      file.onload = function (e: any): void {
        readChunk(remainder ?? '', e.target.result, offset + chunkSize);
      };
      file.readAsText(nextChunk);
    }
    readChunk('', '', 0);
  });
}
/**
 * Since proto-txt doesn't explicitly say whether an attribute is repeated
 * (an array) or not, we keep a hard-coded list of attributes that are known
 * to be repeated. This list is used in parsing time to convert repeated
 * attributes into arrays even when the attribute only shows up once in the
 * object.
 * Repeated fields have to be in sync with graph.proto and all of its
 * dependencies.
 */
const GRAPH_REPEATED_FIELDS: {
  [attrPath: string]: boolean;
} = {
  'library.function': true,
  'library.function.node_def': true,
  'library.function.node_def.input': true,
  'library.function.node_def.attr': true,
  'library.function.node_def.attr.value.list.b': true,
  'library.function.node_def.attr.value.list.f': true,
  'library.function.node_def.attr.value.list.func': true,
  'library.function.node_def.attr.value.list.i': true,
  'library.function.node_def.attr.value.list.s': true,
  'library.function.node_def.attr.value.list.shape': true,
  'library.function.node_def.attr.value.list.shape.dim': true,
  'library.function.node_def.attr.value.list.tensor': true,
  'library.function.node_def.attr.value.list.type': true,
  'library.function.node_def.attr.value.shape.dim': true,
  'library.function.node_def.attr.value.tensor.string_val': true,
  'library.function.node_def.attr.value.tensor.tensor_shape.dim': true,
  'library.function.signature.input_arg': true,
  'library.function.signature.output_arg': true,
  'library.versions': true,
  node: true,
  'node.input': true,
  'node.attr.value.list.b': true,
  'node.attr.value.list.f': true,
  'node.attr.value.list.func': true,
  'node.attr.value.list.i': true,
  'node.attr.value.list.s': true,
  'node.attr.value.list.shape': true,
  'node.attr.value.list.shape.dim': true,
  'node.attr.value.list.tensor': true,
  'node.attr.value.list.type': true,
  'node.attr.value.shape.dim': true,
  'node.attr.value.tensor.string_val': true,
  'node.attr.value.tensor.tensor_shape.dim': true,
};
const METADATA_REPEATED_FIELDS: {
  [attrPath: string]: boolean;
} = {
  'step_stats.dev_stats': true,
  'step_stats.dev_stats.node_stats': true,
  'step_stats.dev_stats.node_stats.output': true,
  'step_stats.dev_stats.node_stats.memory': true,
  'step_stats.dev_stats.node_stats.output.tensor_description.shape.dim': true,
};
/**
 * Parses an ArrayBuffer of a proto txt file into a raw Graph object.
 */
export function parseGraphPbTxt(input: ArrayBuffer): Promise<tf_graph_proto.GraphDef> {
  return parsePbtxtFile(input, GRAPH_REPEATED_FIELDS);
}
/**
 * Parses an ArrayBuffer of a proto txt file into a StepStats object.
 */
export function parseStatsPbTxt(input: ArrayBuffer): Promise<tf_graph_proto.StepStats> {
  return parsePbtxtFile(input, METADATA_REPEATED_FIELDS).then((obj) => obj.step_stats);
}
/**
 * Parses a ArrayBuffer of a proto txt file into javascript object.
 *
 * @param input The ArrayBuffer or file object implementing slice.
 * @param repeatedFields Map (Set) of all the repeated fields, since you can't
 *   tell directly from the pbtxt if a field is repeated or not.
 * @returns The parsed object.
 */
function parsePbtxtFile(
  input: ArrayBuffer,
  repeatedFields: {
    [attrPath: string]: boolean;
  },
): Promise<any> {
  let output: {
    [name: string]: any;
  } = {};
  let stack: Array<{ [name: string]: any }> = [];
  let path: string[] = [];
  let current: {
    [name: string]: any;
  } = output;
  function splitNameAndValueInAttribute(line: string): { name: string; value: any } {
    let colonIndex = line.indexOf(':');
    let name = line.substring(0, colonIndex).trim();
    let value: any = parseValue(line.substring(colonIndex + 2).trim());
    if (name === 'input_data' || name === 'output_data') {
      value = safeJSONParse((value as string).replace(/'{/g, '{').replace(/}'/g, '}').replace(/'/g, '"')) as object;
    } else if (name === 'matched_node_link') {
      value = safeJSONParse((value as string).replace(/'/g, '"')) as string[];
    } else if (name === 'subnodes') {
      value = safeJSONParse((value as string).replace(/'/g, '"')) as string[];
    } else if (name === 'suggestions') {
      value = safeJSONParse((value as string).replace(/'{/g, '{').replace(/}'/g, '}').replace(/'/g, '"')) as object;
    } else {
    }
    if (name === 'attr') {
      const valueObj = safeJSONParse((value as string).replace(/'/g, '"')) as object;
      value = Object.keys(valueObj).map((key) => {
        return {
          key,
          value: valueObj[key],
        };
      });
    }
    return {
      name: name,
      value: value,
    };
  }
  /**
   * Adds a value, given the attribute name and the host object. If the
   * attribute already exists, but is not an array, it will convert it to an
   * array of values.
   *
   * @param obj The host object that holds the attribute.
   * @param name The attribute name (key).
   * @param value The attribute value.
   * @param pathAtt A path that identifies the attribute. Used to check if
   *     an attribute is an array or not.
   */
  function addAttribute(
    obj: { [name: string]: any },
    name: string,
    value: { [name: string]: any } | string | number | boolean,
    pathAtt: string[],
  ): void {
    // We treat 'node' specially since it is done so often.
    let existingValue = obj[name];
    if (existingValue == null) {
      obj[name] = pathAtt.join('.') in repeatedFields ? [value] : value;
    } else if (Array.isArray(existingValue)) {
      existingValue.push(value);
    } else {
      obj[name] = [existingValue, value];
    }
  }
  // Run through the file a line at a time.
  return streamParse(input, (line: string) => {
    let lineNew = line.trim();
    if (!lineNew) {
      return;
    }
    switch (lineNew[lineNew.length - 1]) {
      case '{': {
        // create new object
        let name = lineNew.substring(0, lineNew.length - 2).trim();
        let newValue: {
          [name: string]: any;
        } = {};
        stack.push(current);
        path.push(name);
        addAttribute(current, name, newValue, path);
        current = newValue;
        break;
      }
      case '}': {
        current = stack.pop() ?? {};
        path.pop();
        break;
      }
      default: {
        let x = splitNameAndValueInAttribute(lineNew);
        addAttribute(current, x.name, x.value, path.concat(x.name));
        break;
      }
    }
  }).then(() => {
    return output;
  });
}
