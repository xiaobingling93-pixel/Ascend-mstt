/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

export async function setup(): Promise<void> {
  await google.charts.load('current', {
    packages: ['corechart', 'table', 'timeline'],
  });
}
