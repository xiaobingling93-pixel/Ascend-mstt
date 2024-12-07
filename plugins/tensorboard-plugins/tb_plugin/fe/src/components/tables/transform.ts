/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { CallStackTableData, CallStackTableDataInner } from '../../api';

export interface CallStackFrame {
  file?: string;
  line?: number;
  raw: string;
}

export interface TransformedCallStackDataInner extends CallStackTableDataInner {
  callStackFrames: CallStackFrame[];
}

const lineRegex = /\([0-9]+\)$/;

function parseCallStackLine(raw: string): CallStackFrame {
  let rawResult = raw.trim();
  const results = rawResult.split(':');
  const location = results.slice(0, results.length - 1).join(':');

  const result = lineRegex.exec(location);
  if (!result) {
    return { raw: rawResult };
  }

  const lineWithParens = result[0].trim();
  const file = rawResult.slice(0, result.index).trim();
  const line = Number(
    lineWithParens.substr(1, lineWithParens.length - 2).trim()
  );

  return {
    raw: rawResult,
    file,
    line,
  };
}

function parseCallStack(callStack?: string): CallStackFrame[] {
  const lines = (callStack ?? '')
    .trim()
    .split(';')
    .map((x) => x.trim());
  return lines.map(parseCallStackLine);
}

function transformCallStackData(
  data: CallStackTableDataInner
): TransformedCallStackDataInner {
  return {
    ...data,
    callStackFrames: parseCallStack(data.call_stack),
  };
}

export function transformTableData(
  data: CallStackTableData
): TransformedCallStackDataInner[] {
  return data.map(transformCallStackData);
}
