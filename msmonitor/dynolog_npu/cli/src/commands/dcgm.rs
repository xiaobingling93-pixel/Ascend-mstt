// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use crate::DynoClient;
use super::utils;

// This module contains the handling logic for dcgm

/// Pause dcgm module profiling
pub fn run_dcgm_pause(
    mut client: DynoClient,
    duration_s: i32,
) -> Result<()> {
    let msg = format!(r#"{{"fn":"dcgmPause", "duration_s":{}}}"#, duration_s);
    utils::send_msg(&mut client, &msg)?;
    let resp_str = utils::get_resp(&mut client)?;
    println!("{}", resp_str);
    Ok(())
}

/// Resume dcgm module profiling
pub fn run_dcgm_resume(
    mut client: DynoClient,
) -> Result<()> {
    utils::send_msg(&mut client, r#"{"fn":"dcgmResume"}"#)?;
    let resp_str = utils::get_resp(&mut client)?;
    println!("{}", resp_str);
    Ok(())
}