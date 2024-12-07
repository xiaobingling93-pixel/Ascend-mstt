/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export function isDef<T>(v?: T | null): v is T {
  return v !== null && v !== undefined;
}

export function assertDef<T>(v?: T | null): asserts v is T {
  if (!isDef(v)) {
    throw new Error('Must be defined');
  }
}

export function firstOrUndefined<T>(v?: T[]): T | undefined {
  if (!v || !v.length) {
    return undefined;
  }
  return v[0];
}
