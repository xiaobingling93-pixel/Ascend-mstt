// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use std::io::{Read, Write};

use anyhow::Result;

use crate::DynoClient;

pub fn send_msg(client: &mut DynoClient, msg: &str) -> Result<()> {
    match client {
        DynoClient::Secure(secure_client) => {
            let msg_len: [u8; 4] = i32::try_from(msg.len()).unwrap().to_ne_bytes();
            secure_client.write_all(&msg_len)?;
            secure_client.write_all(msg.as_bytes())?;
            secure_client.flush()?;
        }
        DynoClient::Insecure(insecure_client) => {
            let msg_len: [u8; 4] = i32::try_from(msg.len()).unwrap().to_ne_bytes();
            insecure_client.write_all(&msg_len)?;
            insecure_client.write_all(msg.as_bytes())?;
            insecure_client.flush()?;
        }
    }
    Ok(())
}

pub fn get_resp(client: &mut DynoClient) -> Result<String> {
    let mut len_buf = [0u8; 4];
    let mut resp_buf;
    
    match client {
        DynoClient::Secure(secure_client) => {
            secure_client.read_exact(&mut len_buf)?;
            let len = u32::from_ne_bytes(len_buf) as usize;
            resp_buf = vec![0u8; len];
            secure_client.read_exact(&mut resp_buf)?;
        }
        DynoClient::Insecure(insecure_client) => {
            insecure_client.read_exact(&mut len_buf)?;
            let len = u32::from_ne_bytes(len_buf) as usize;
            resp_buf = vec![0u8; len];
            insecure_client.read_exact(&mut resp_buf)?;
        }
    }
    
    Ok(String::from_utf8(resp_buf)?)
}