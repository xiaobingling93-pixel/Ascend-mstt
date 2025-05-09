use rustls::{ClientConnection, StreamOwned};
use std::net::TcpStream;

use anyhow::Result;

#[path = "utils.rs"]
mod utils;

#[derive(Debug)]
pub struct NpuMonitorConfig {
    pub npu_monitor_start: bool,
    pub npu_monitor_stop: bool,
    pub report_interval_s: u32,
    pub mspti_activity_kind: String,
}

impl NpuMonitorConfig {
    fn config(&self) -> String {
        format!(
            r#"
NPU_MONITOR_START={}
NPU_MONITOR_STOP={}
REPORT_INTERVAL_S={}
MSPTI_ACTIVITY_KIND={}"#,
            self.npu_monitor_start,
            self.npu_monitor_stop,
            self.report_interval_s,
            self.mspti_activity_kind
        )
    }
}

pub fn run_npumonitor(
    mut client: StreamOwned<ClientConnection, TcpStream>,
    config: NpuMonitorConfig,
) -> Result<()> {
    let config_str = config.config();
    println!("Npu monitor config = \n{}", config_str);
    let config_str = config_str.replace('\n', "\\n");

    let request_json = format!(
        r#"
{{
    "fn": "setKinetOnDemandRequest",
    "config": "{}",
    "job_id": 0,
    "pids": [0],
    "process_limit": 3
}}"#,
        config_str
    );

    utils::send_msg(&mut client, &request_json).expect("Error sending message to service");

    let resp_str = utils::get_resp(&mut client).expect("Unable to decode output bytes");

    println!("response = {}", resp_str);

    Ok(())
}
