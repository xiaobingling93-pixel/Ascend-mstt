/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react';
import debounce from '@material-ui/core/utils/debounce';

export function useResizeEventDependency(): readonly [number] {
  const [version, setVersion] = React.useState(0);

  const increaseVersion = React.useCallback(
    debounce(() => {
      setVersion((prev) => prev + 1);
    }, 100),
    []
  );

  React.useEffect(() => {
    window.addEventListener('resize', increaseVersion);

    return (): void => {
      window.removeEventListener('resize', increaseVersion);
    };
  }, []);

  return [version] as const;
}
