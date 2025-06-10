// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use crate::DynoClient;
use super::utils;

// This module contains the handling logic for querying dyno version

/// Get version info
pub fn run_version(mut client: DynoClient) -> Result<()> {
    utils::send_msg(&mut client, r#"{"fn":"getVersion"}"#)?;
    let resp_str = utils::get_resp(&mut client)?;
    println!("{}", resp_str);
    Ok(())
}