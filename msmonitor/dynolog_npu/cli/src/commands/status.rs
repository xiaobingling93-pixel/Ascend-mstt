// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use crate::DynoClient;
use super::utils;

pub fn run_status(mut client: DynoClient) -> Result<()> {
    utils::send_msg(&mut client, r#"{"fn":"getStatus"}"#)?;
    let resp_str = utils::get_resp(&mut client)?;
    println!("{}", resp_str);
    Ok(())
}