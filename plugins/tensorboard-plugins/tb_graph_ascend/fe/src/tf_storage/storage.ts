/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
import * as _ from 'lodash';
import {
  addHashListener,
  addStorageListener,
  fireStorageChanged,
  ListenKey,
  removeHashListenerByKey,
  removeStorageListenerByKey,
} from './listeners';
import {
  componentToDict,
  dictToComponent,
  readComponent,
  TAB_KEY,
  unsetFromURI,
  updateUrlDict,
  writeComponent,
} from './storage_utils';

export { getUrlDict as getUrlHashDict } from './storage_utils';

/**
 * The name of the property for users to set on a Polymer component
 * in order for its stored properties to be stored in the URI unambiguously.
 * (No need to set this if you want multiple instances of the component to
 * share URI state)
 *
 * Example:
 * <my-component disambiguator="0"></my-component>
 *
 * The disambiguator should be set to any unique value so that multiple
 * instances of the component can store properties in URI storage.
 *
 * Because it's hard to dereference this variable in HTML property bindings,
 * it is NOT safe to change the disambiguator string without find+replace
 * across the codebase.
 */
export const DISAMBIGUATOR = 'disambiguator';

export const {
  get: getString,
  set: setString,
  getInitializer: getStringInitializer,
  getObserver: getStringObserver,
  disposeBinding: disposeStringBinding,
} = makeBindings(
  (x) => x,
  (x) => x,
);
export const {
  get: getBoolean,
  set: setBoolean,
  getInitializer: getBooleanInitializer,
  getObserver: getBooleanObserver,
  disposeBinding: disposeBooleanBinding,
} = makeBindings(
  (s) => {
    if (s === 'true') {
      return true;
    } else if (s === 'false') {
      return false;
    } else {
      return undefined;
    }
  },
  (b) => b.toString(),
);
export const {
  get: getNumber,
  set: setNumber,
  getInitializer: getNumberInitializer,
  getObserver: getNumberObserver,
  disposeBinding: disposeNumberBinding,
} = makeBindings(
  (s) => Number(s),
  (n) => n.toString(),
);
export const {
  get: getObject,
  set: setObject,
  getInitializer: getObjectInitializer,
  getObserver: getObjectObserver,
  disposeBinding: disposeObjectBinding,
} = makeBindings(
  (s) => JSON.parse(atob(s)) as Record<string, string>,
  (o) => btoa(JSON.stringify(o)),
);
export interface StorageOptions<T> {
  defaultValue?: T;
  useLocalStorage?: boolean;
}
export interface AutoStorageOptions<T> extends StorageOptions<T> {
  polymerProperty?: string;
}
export interface SetterOptions<T> extends StorageOptions<T> {
  defaultValue?: T;
  useLocalStorage?: boolean;
  useLocationReplace?: boolean;
}
export function makeBindings<T>(
  fromString: (string) => T,
  toString: (T) => string,
): {
  get: (key: string, option?: StorageOptions<T>) => T;
  set: (key: string, value: T, option?: SetterOptions<T>) => void;
  getInitializer: (key: string, options: AutoStorageOptions<T>) => () => T;
  getObserver: (key: string, options: AutoStorageOptions<T>) => () => void;
  disposeBinding: () => void;
} {
  const hashListeners: ListenKey[] = [];
  const storageListeners: ListenKey[] = [];
  function get(key: string, options: StorageOptions<T> = {}): T {
    const { defaultValue, useLocalStorage = false } = options;
    const value = useLocalStorage ? window.localStorage.getItem(key) : componentToDict(readComponent())[key];
    return (value === undefined ? _.cloneDeep(defaultValue) : fromString(value)) as T;
  }
  function set(key: string, value: T, options: SetterOptions<T> = {}): void {
    const { defaultValue, useLocalStorage = false, useLocationReplace = false } = options;
    const stringValue = toString(value);
    if (useLocalStorage) {
      window.localStorage.setItem(key, stringValue);
      // Because of listeners.ts:[1], we need to manually notify all UI elements
      // listening to storage within the tab of a change.
      fireStorageChanged();
    } else if (!_.isEqual(value, get(key, { useLocalStorage }))) {
      if (_.isEqual(value, defaultValue)) {
        unsetFromURI(key, useLocationReplace);
      } else {
        const items = componentToDict(readComponent());
        items[key] = stringValue;
        writeComponent(dictToComponent(items), useLocationReplace);
      }
    }
  }
  /**
   * Returns a function that can be used on a `value` declaration to a Polymer
   * property. It updates the `polymerProperty` when storage changes -- i.e.,
   * when `useLocalStorage`, it listens to storage change from another tab and
   * when `useLocalStorage=false`, it listens to hashchange.
   */
  function getInitializer(key: string, options: StorageOptions<T>): () => T {
    const fullOptions = {
      defaultValue: options.defaultValue,
      polymerProperty: key,
      useLocalStorage: false,
      ...options,
    };
    return function () {
      const uriStorageName = getURIStorageName(this, key);
      // setComponentValue will be called every time the underlying storage
      // changes and is responsible for ensuring that new state will propagate
      // to the component with specified property. It is important that this
      // function does not re-assign needlessly, to avoid Polymer observer
      // churn.
      const setComponentValue = (): void => {
        const storedValue = get(uriStorageName, fullOptions);
        const currentValue = this[fullOptions.polymerProperty];
        if (!_.isEqual(storedValue, currentValue)) {
          this[fullOptions.polymerProperty] = storedValue;
        }
      };
      const addListener = fullOptions.useLocalStorage ? addStorageListener : addHashListener;
      const listenKey = addListener(() => setComponentValue());
      if (fullOptions.useLocalStorage) {
        storageListeners.push(listenKey);
      } else {
        hashListeners.push(listenKey);
      }
      // Set the value on the property.
      setComponentValue();
      return this[fullOptions.polymerProperty];
    };
  }
  function disposeBinding(): void {
    hashListeners.forEach((key) => removeHashListenerByKey(key));
    storageListeners.forEach((key) => removeStorageListenerByKey(key));
  }
  function getObserver(key: string, options: StorageOptions<T>): () => void {
    const fullOptions = {
      defaultValue: options.defaultValue,
      polymerProperty: key,
      useLocalStorage: false,
      ...options,
    };
    return function () {
      const uriStorageName = getURIStorageName(this, key);
      const newVal = this[fullOptions.polymerProperty];
      set(uriStorageName, newVal, fullOptions);
    };
  }
  return { get, set, getInitializer, getObserver, disposeBinding };
}
/**
 * Get a unique storage name for a (Polymer component, propertyName) tuple.
 *
 * DISAMBIGUATOR must be set on the component, if other components use the
 * same propertyName.
 */
function getURIStorageName(component: Record<string, unknown>, propertyName: string): string {
  const d = component[DISAMBIGUATOR];
  const components = d == null ? [propertyName] : [d, propertyName];
  return components.join('.');
}
