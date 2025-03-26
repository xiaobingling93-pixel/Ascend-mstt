/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
/**
 * Compares tag names asciinumerically broken into components.
 *
 * <p>This is the comparison function used for sorting most string values in
 * TensorBoard. Unlike the standard asciibetical comparator, this function
 * knows that 'a10b' > 'a2b'. Fixed point and engineering notation are
 * supported. This function also splits the input by slash and underscore to
 * perform array comparison. Therefore it knows that 'a/a' < 'a+/a' even
 * though '+' < '/' in the ASCII table.
 */
export function compareTagNames(a: string, b: string): number {
  let ai = 0;
  let bi = 0;
  while (true) {
    // Handle end of strings
    if (ai === a.length) {
      return bi === b.length ? 0 : -1;
    }
    if (bi === b.length) {
      return 1;
    }

    // Check for digits
    if (isDigit(a[ai]) && isDigit(b[bi])) {
      const ais = ai;
      const bis = bi;
      
      // Consume all consecutive digits (simplified from original)
      while (ai < a.length && isDigit(a[ai])) ai++;
      while (bi < b.length && isDigit(b[bi])) bi++;
      
      const an = parseInt(a.slice(ais, ai), 10);
      const bn = parseInt(b.slice(bis, bi), 10);
      
      if (an !== bn) {
        return an - bn;
      }
      continue;
    }

    // Treat underscore as regular character (not a break)
    if (a[ai] === '_' && b[bi] === '_') {
      ai++;
      bi++;
      continue;
    }

    // Simplified break character handling
    if (isBreak(a[ai]) && !isBreak(b[bi])) {
      return -1;
    }
    if (!isBreak(a[ai]) && isBreak(b[bi])) {
      return 1;
    }

    // Regular character comparison
    if (a[ai] < b[bi]) {
      return -1;
    }
    if (a[ai] > b[bi]) {
      return 1;
    }

    ai++;
    bi++;
  }
}

// Simplified digit check
function isDigit(c: string): boolean {
  return c >= '0' && c <= '9';
}

// Simplified break character check
function isBreak(c: string): boolean {
  return c === '/'; // Only treat slash as break character
}
