/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import debounce from '@material-ui/core/utils/debounce';
import * as React from 'react';

export enum UseTop {
  NotUse = 'NotUse',
  Use = 'Use',
}

interface IOptions {
  defaultTop?: number;
  defaultUseTop?: UseTop;
  noDebounce?: boolean;
  wait?: number;
}

export function useTopN(options?: IOptions) {
  let realOptions = options ?? {};

  const [topText, setTopText] = React.useState(
    String(realOptions.defaultTop ?? 15)
  );
  const [actualTop, setActualTop] = React.useState<number | undefined>(
    Number(topText)
  );
  const [useTop, setUseTop] = React.useState(
    realOptions.defaultUseTop ?? UseTop.NotUse
  );

  const setActualDebounce = !realOptions.noDebounce
    ? React.useCallback(debounce(setActualTop, realOptions.wait ?? 500), [])
    : setActualTop;
  React.useEffect(() => {
    if (useTop !== UseTop.Use) {
      setActualDebounce(undefined);
    } else if (topIsValid(topText)) {
      setActualDebounce(Number(topText));
    } else {
      setActualDebounce(actualTop);
    }
  }, [topText, useTop]);

  return [topText, actualTop, useTop, setTopText, setUseTop] as const;
}

export function topIsValid(topText: string) {
  const top = Number(topText);
  return !Number.isNaN(top) && top > 0 && Number.isInteger(top);
}
