/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export function navToCode(filename: string, line: number): void {
  window.parent.parent.postMessage(
    {
      filename,
      line,
    },
    window.origin
  );
}
