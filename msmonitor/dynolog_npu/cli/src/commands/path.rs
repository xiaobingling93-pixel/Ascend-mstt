// Copyright (c) Meta Platforms, Inc. and affiliates.
// Copyright (c) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use std::ffi::CString;
use std::path::Path;
use libc::{R_OK, W_OK, X_OK};

const MAX_PATH_SIZE: usize = 1024;
const DIR_CHECK_MODE: i32 = R_OK | W_OK | X_OK;

const INVALID_CHAR: &[(&str, &str)] = &[
    ("\n", "\\n"),
    ("\u{000C}", "\\f"),
    ("\r", "\\r"),
    ("\u{0008}", "\\b"),
    ("\t", "\\t"),
    ("\u{000B}", "\\v"),
    ("\u{007F}", "\\u007F"),
    ("\"", "\\\""),
    ("'", "'"),
    ("\\", "\\\\"),
    ("%", "\\%"),
    (">", "\\>"),
    ("<", "\\<"),
    ("|", "\\|"),
    ("&", "\\&"),
    ("$", "\\$"),
];

fn rstrip(s: &str, chars: &str) -> String {
    s.trim_end_matches(|c| chars.contains(c)).to_string()
}

pub struct PathUtils;

impl PathUtils {
    pub fn access(path: &str, mode: i32) -> bool {
        if path.is_empty() {
            println!("ERROR: The file path is empty.");
            return false;
        }

        let c_path = match CString::new(path) {
            Ok(p) => p,
            Err(_) => {
                println!("ERROR: Invalid path (contains null byte): {}", path);
                return false;
            }
        };

        unsafe {
            libc::access(c_path.as_ptr(), mode) == 0
        }
    }

    /// 检查文件或目录是否存在
    pub fn exist(path: &str) -> bool {
        Path::new(path).exists()
    }

    /// 判断是否为软链接
    pub fn is_soft_link(path: &str) -> bool {
        if path.is_empty() {
            println!("ERROR: The file path is empty.");
            return false;
        }

        let trimmed = rstrip(path, "./");
        let p = Path::new(&trimmed);

        match p.symlink_metadata() {
            Ok(metadata) => metadata.file_type().is_symlink(),
            Err(e) => {
                println!("ERROR: The file lstat failed: {}", e);
                false
            }
        }
    }

    pub fn is_file(path: &str) -> bool {
        if path.is_empty() {
            println!("ERROR: The file path is empty.");
            return false;
        }

        let p = Path::new(path);
        match p.metadata() {
            Ok(metadata) => metadata.is_file(),
            Err(e) => {
                println!("ERROR: The file stat failed: {}", e);
                false
            }
        }
    }

    pub fn check_dir(path: &str, should_exist: bool) -> bool {
        if path.is_empty() {
            println!("ERROR: The path is empty.");
            return false;
        }

        if path.len() > MAX_PATH_SIZE {
            println!("ERROR: The length of path is too long, max allowed: {}", MAX_PATH_SIZE);
            return false;
        }

        for &(invalid, _) in INVALID_CHAR {
            if path.contains(invalid) {
                println!("ERROR: The path contains invalid character: {:?}", invalid);
                return false;
            }
        }

        if !Self::exist(path) {
            if should_exist {
                println!("ERROR: The path does not exist: {}", path);
                return false;
            } else {
                return true;
            }
        }

        if Self::is_file(path) {
            println!("ERROR: The path is a file: {}", path);
            return false;
        }

        if Self::is_soft_link(path) {
            println!("ERROR: The path is a soft link: {}", path);
            return false;
        }

        if !Self::access(path, DIR_CHECK_MODE) {
            println!("ERROR: The path has no rwx access: {}", path);
            return false;
        }

        true
    }
}