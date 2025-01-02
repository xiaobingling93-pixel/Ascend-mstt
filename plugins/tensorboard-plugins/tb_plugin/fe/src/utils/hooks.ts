/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';

const cbs: Array<() => void> = [];
export const useOnResize = (cb: () => void): void => {
  React.useEffect(() => {
    if (cbs.length === 0) {
      window.addEventListener('resize', () => {
        cbs.forEach((callback) => callback());
      });
    }
    cbs.push(cb);

    return (): void => {
      const idx = cbs.findIndex(cb);
      if (idx > -1) {
        cbs.splice(idx, 1);
      }
      if (cbs.length === 0) {
        window.removeEventListener('reset', cb);
      }
    };
  }, [cb]);
};
