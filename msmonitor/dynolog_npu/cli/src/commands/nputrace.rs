use anyhow::Result;
use serde_json::Value;
use crate::DynoClient;
use super::utils;

#[derive(Debug)]
pub enum NpuTraceTriggerConfig {
    DurationBased {
        profile_start_time: u64,
        duration_ms: u64,
    },
    IterationBased {
        start_step: u64,
        iterations: i64,
    },
}

impl NpuTraceTriggerConfig {
    fn config(&self) -> String {
        match *self {
            NpuTraceTriggerConfig::DurationBased {
                profile_start_time,
                duration_ms,
            } => format!(
                "PROFILE_START_TIME={}\nACTIVITIES_DURATION_MSECS={}",
                profile_start_time, duration_ms
            ),
            NpuTraceTriggerConfig::IterationBased {
                start_step,
                iterations,
            } => format!(
                r#"PROFILE_START_ITERATION=0
PROFILE_START_STEP={}
ACTIVITIES_ITERATIONS={}"#,
                start_step, iterations
            ),
        }
    }
}

// torch npu profiler config
#[derive(Debug)]
pub struct NpuTraceOptions {
    pub record_shapes: bool,
    pub profile_memory: bool,
    pub with_stack: bool,
    pub with_flops: bool,
    pub with_modules: bool,
    pub activities: String,
    pub analyse: bool,
    pub profiler_level: String,
    pub aic_metrics: String,
    pub l2_cache: bool,
    pub op_attr: bool,
    pub msprof_tx: bool,
    pub gc_detect_threshold: Option<f32>,
    pub data_simplification: String,
    pub export_type: String,
    pub host_sys: String,
    pub sys_io: bool,
    pub sys_interconnection: bool,
    pub mstx_domain_include: Option<String>,
    pub mstx_domain_exclude: Option<String>,
}

impl NpuTraceOptions {
    fn config(&self) -> String {
        format!(
            r#"
PROFILE_RECORD_SHAPES={}
PROFILE_PROFILE_MEMORY={}
PROFILE_WITH_STACK={}
PROFILE_WITH_FLOPS={}
PROFILE_WITH_MODULES={}
PROFILE_ACTIVITIES={}
PROFILE_ANALYSE={}
PROFILE_PROFILER_LEVEL={}
PROFILE_AIC_METRICS={}
PROFILE_L2_CACHE={}
PROFILE_OP_ATTR={}
PROFILE_MSPROF_TX={}
PROFILE_GC_DETECT_THRESHOLD={}
PROFILE_DATA_SIMPLIFICATION={}
PROFILE_EXPORT_TYPE={}
PROFILE_HOST_SYS={}
PROFILE_SYS_IO={}
PROFILE_SYS_INTERCONNECTION={}
PROFILE_MSTX_DOMAIN_INCLUDE={}
PROFILE_MSTX_DOMAIN_EXCLUDE={}"#,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            self.activities,
            self.analyse,
            self.profiler_level,
            self.aic_metrics,
            self.l2_cache,
            self.op_attr,
            self.msprof_tx,
            self.gc_detect_threshold.map_or("None".to_string(), |v| v.to_string()),
            self.data_simplification,
            self.export_type,
            self.host_sys,
            self.sys_io,
            self.sys_interconnection,
            self.mstx_domain_include.clone().map_or("None".to_string(), |v| v.to_string()),
            self.mstx_domain_exclude.clone().map_or("None".to_string(), |v| v.to_string())
        )
    }
}

#[derive(Debug)]
pub struct NpuTraceConfig {
    pub log_file: String,
    pub trigger_config: NpuTraceTriggerConfig,
    pub trace_options: NpuTraceOptions,
}

impl NpuTraceConfig {
    fn config(&self) -> String {
        format!(
            "ACTIVITIES_LOG_FILE={}\n{}{}",
            self.log_file,
            self.trigger_config.config(),
            self.trace_options.config()
        )
    }
}

pub fn run_nputrace(
    mut client: DynoClient,
    job_id: u64,
    pids: &str,
    process_limit: u32,
    config: NpuTraceConfig,
) -> Result<()> {
    let config_str = config.config();
    println!("NpuTrace config = \n{}", config_str);
    let config_str = config_str.replace('\n', "\\n");

    let request_json = format!(
        r#"
{{
    "fn": "setKinetOnDemandRequest",
    "config": "{}",
    "job_id": {},
    "pids": [{}],
    "process_limit": {}
}}"#,
        config_str, job_id, pids, process_limit
    );

    utils::send_msg(&mut client, &request_json)?;

    let resp_str = utils::get_resp(&mut client)?;

    println!("response = {}", resp_str);

    let resp_v: Value = serde_json::from_str(&resp_str)?;
    let processes = resp_v["processesMatched"].as_array().unwrap();

    if processes.is_empty() {
        println!("No processes were matched, please check --job-id or --pids flags");
    } else {
        println!("Matched {} processes", processes.len());
        println!("Trace output files will be written to:");

        for pid in processes {
            let pid = pid.as_i64().unwrap();
            println!(
                "    {}",
                config.log_file.replace(".json", &format!("_{}.json", pid))
            );
        }
    }

    Ok(())
}


#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn test_nputrace_trigger_config() {
        let trigger_config = NpuTraceTriggerConfig::DurationBased {
            profile_start_time: 1000,
            duration_ms: 1000,
        };
        assert_eq!(
            trigger_config.config(),
            r#"PROFILE_START_TIME=1000
ACTIVITIES_DURATION_MSECS=1000"#
        );

        let trigger_config = NpuTraceTriggerConfig::IterationBased {    
            profile_start_step: 1000,
            iterations: 1000,
        };
        assert_eq!(
            trigger_config.config(),
            r#"PROFILE_START_ITERATION=0
PROFILE_START_STEP=1000
ACTIVITIES_ITERATIONS=1000"#
        );
    }

    #[test]
    fn test_nputrace_config() {
        let config = NpuTraceConfig {
            log_file: "test.json".to_string(),
            trigger_config: NpuTraceTriggerConfig::DurationBased {
                profile_start_time: 1000,
                duration_ms: 1000,
            },
            trace_options: NpuTraceOptions {
                record_shapes: true,
                profile_memory: false,
                with_stack: true,
                with_flops: true,
                with_modules: true,
                activities: "CPU,NPU".to_string(),
                analyse: false,
                profiler_level: "Level0".to_string(),
                aic_metrics: "AiCoreNone".to_string(),
                l2_cache: true,
                op_attr: true,
                msprof_tx: true,
                gc_detect_threshold: 0.1,
                data_simplification: "true",
                export_type: "Text".to_string(),
                host_sys: "cpu".to_string(),
                sys_io: true,
                sys_interconnection: true,
                mstx_domain_include: "domain1".to_string(),
                mstx_domain_exclude: "domain2".to_string(),
            },
        };
        assert_eq!(
            config.config(),
            r#"ACTIVITIES_LOG_FILE=test.json
PROFILE_START_TIME=1000
ACTIVITIES_DURATION_MSECS=1000
PROFILE_RECORD_SHAPES=true
PROFILE_PROFILE_MEMORY=false
PROFILE_WITH_STACK=true
PROFILE_WITH_FLOPS=true
PROFILE_WITH_MODULES=true
PROFILE_ACTIVITIES=CPU,NPU
PROFILE_ANALYSE=false
PROFILE_PROFILER_LEVEL=Level0
PROFILE_AIC_METRICS=AiCoreNone
PROFILE_L2_CACHE=true
PROFILE_OP_ATTR=true
PROFILE_MSPROF_TX=true
PROFILE_GC_DETECT_THRESHOLD=0.1
PROFILE_DATA_SIMPLIFICATION=true
PROFILE_EXPORT_TYPE=Text
PROFILE_HOST_SYS=cpu
PROFILE_SYS_IO=true
PROFILE_SYS_INTERCONNECTION=true
PROFILE_MSTX_DOMAIN_INCLUDE=domain1
PROFILE_MSTX_DOMAIN_EXCLUDE=domain2"#
        );
    }
}
