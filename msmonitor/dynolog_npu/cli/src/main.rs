// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use std::fs::File;
use std::io::BufReader;
use rustls::{Certificate, RootCertStore, PrivateKey, ClientConnection, StreamOwned};
use std::sync::Arc;
use std::net::TcpStream;
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::io;

use anyhow::Result;
use clap::Parser;
use std::collections::HashSet;

// Make all the command modules accessible to this file.
mod commands;
use commands::gputrace::GpuTraceConfig;
use commands::gputrace::GpuTraceOptions;
use commands::gputrace::GpuTraceTriggerConfig;
use commands::nputrace::NpuTraceConfig;
use commands::nputrace::NpuTraceOptions;
use commands::nputrace::NpuTraceTriggerConfig;
use commands::npumonitor::NpuMonitorConfig;
use commands::*;

/// Instructions on adding a new Dyno CLI command:
///
/// 1. Add a new variant to the `Command` enum.
///    Please include a description of the command and, if applicable, its flags/subcommands.
///
/// 2. Create a new file for the command's implementation in the commands/ directory (ie
///    commands/status.rs). This new file is where the command should be implemented.
///    Make the new command's module accessible from this file by adding
///    a new line with `pub mod <newfile>;` to commands/mod.rs.
///
///
/// 3. Add a branch to the match statement in main() to handle the new enum variant (from step 1).
///    From here, invoke the handling logic defined in the new file (from step 2). In an effort to keep
///    the command dispatching logic clear and concise, please keep the code in the match branch to a minimum.

const DYNO_PORT: u16 = 1778;

#[derive(Debug, Parser)]
struct Opts {
    #[clap(long, default_value = "localhost")]
    hostname: String,
    #[clap(long, default_value_t = DYNO_PORT)]
    port: u16,
    #[clap(long, required = true)]
    certs_dir: String,
    #[clap(subcommand)]
    cmd: Command,
}

const ALLOWED_VALUES: &[&str] = &["Marker", "Kernel", "API", "Hccl", "Memory", "MemSet", "MemCpy"];

fn parse_mspti_activity_kinds(src: &str)  -> Result<String, String>{
    let allowed_values: HashSet<&str> = ALLOWED_VALUES.iter().cloned().collect();

    let kinds: Vec<&str> = src.split(',').map(|s| s.trim()).collect();

    for kind in &kinds {
        if !allowed_values.contains(kind) {
            return Err(format!("Invalid MSPTI activity kind: {}, Possible values: {:?}.]", kind, allowed_values));
        }
    }
    
    Ok(src.to_string())
}

const ALLOWED_HOST_SYSTEM_VALUES: &[&str] = &["cpu", "mem", "disk", "network", "osrt"];

fn parse_host_sys(src: &str) -> Result<String, String>{
    if src == "None" {
        return Ok(src.to_string());
    }

    let allowed_host_sys_values: HashSet<&str> = ALLOWED_HOST_SYSTEM_VALUES.iter().cloned().collect();

    let host_systems: Vec<&str> = src.split(',').map(|s| s.trim()).collect();

    for host_system in &host_systems {
        if !allowed_host_sys_values.contains(host_system) {
            return Err(format!("Invalid NPU Trace host system: {}, Possible values: {:?}.]", host_system,
            allowed_host_sys_values));
        }
    }
    let result = host_systems.join(",");
    Ok(result)
}

#[derive(Debug, Parser)]
enum Command {
    /// Check the status of a dynolog process
    Status,
    /// Check the version of a dynolog process
    Version,
    /// Capture gputrace
    Gputrace {
        /// Job id of the application to trace.
        #[clap(long, default_value_t = 0)]
        job_id: u64,
        /// List of pids to capture trace for (comma separated).
        #[clap(long, default_value = "0")]
        pids: String,
        /// Duration of trace to collect in ms.
        #[clap(long, default_value_t = 500)]
        duration_ms: u64,
        /// Training iterations to collect, this takes precedence over duration.
        #[clap(long, default_value_t = -1)]
        iterations: i64,
        /// Log file for trace.
        #[clap(long)]
        log_file: String,
        /// Unix timestamp used for synchronized collection (milliseconds since epoch).
        #[clap(long, default_value_t = 0)]
        profile_start_time: u64,
        /// Start iteration roundup, starts an iteration based trace at a multiple
        /// of this value.
        #[clap(long, default_value_t = 1)]
        profile_start_iteration_roundup: u64,
        /// Max number of processes to profile.
        #[clap(long, default_value_t = 3)]
        process_limit: u32,
        /// Record PyTorch operator input shapes and types.
        #[clap(long, action)]
        record_shapes: bool,
        /// Profile PyTorch memory.
        #[clap(long, action)]
        profile_memory: bool,
        /// Capture Python stacks in traces.
        #[clap(long, action)]
        with_stacks: bool,
        /// Annotate operators with analytical flops.
        #[clap(long, action)]
        with_flops: bool,
        /// Capture PyTorch operator modules in traces.
        #[clap(long, action)]
        with_modules: bool,
    },
    /// Capture nputrace. Subcommand functions aligned with Ascend Torch Profiler.
    Nputrace {
        /// Job id of the application to trace.
        #[clap(long, default_value_t = 0)]
        job_id: u64,
        /// List of pids to capture trace for (comma separated).
        #[clap(long, default_value = "0")]
        pids: String,
        /// Duration of trace to collect in ms.
        #[clap(long, default_value_t = 500)]
        duration_ms: u64,
        /// Training iterations to collect, this takes precedence over duration.
        #[clap(long, default_value_t = -1)]
        iterations: i64,
        /// Log file for trace.
        #[clap(long)]
        log_file: String,
        /// Unix timestamp used for synchronized collection (milliseconds since epoch).
        #[clap(long, default_value_t = 0)]
        profile_start_time: u64,
        /// Number of steps to start profile.
        #[clap(long, default_value_t = 0)]
        start_step: u64,
        /// Max number of processes to profile.
        #[clap(long, default_value_t = 3)]
        process_limit: u32,
        /// Whether to record PyTorch operator input shapes and types.
        #[clap(long, action)]
        record_shapes: bool,
        /// Whether to profile PyTorch memory.
        #[clap(long, action)]
        profile_memory: bool,
        /// Whether to profile the Python call stack in trace.
        #[clap(long, action)]
        with_stack: bool,
        /// Annotate operators with analytical flops.
        #[clap(long, action)]
        with_flops: bool,
        /// Whether to profile PyTorch operator modules in traces.
        #[clap(long, action)]
        with_modules: bool,
        /// The scope of the profile's events.
        #[clap(long, value_parser = ["CPU,NPU", "NPU,CPU", "CPU", "NPU"], default_value = "CPU,NPU")]
        activities: String,
        /// Profiler level.
        #[clap(long, value_parser = ["Level0", "Level1", "Level2", "Level_none"], default_value = "Level0")]
        profiler_level: String,
        /// AIC metrics.
        #[clap(long, value_parser = ["AiCoreNone", "PipeUtilization", "ArithmeticUtilization", "Memory", "MemoryL0", "ResourceConflictRatio", "MemoryUB", "L2Cache", "MemoryAccess"], default_value = "AiCoreNone")]
        aic_metrics: String,
        /// Whether to analyse the data after collection.
        #[clap(long, action)]
        analyse: bool,
        /// Whether to collect L2 cache.
        #[clap(long, action)]
        l2_cache: bool,
        /// Whether to collect op attributes.
        #[clap(long, action)]
        op_attr: bool,
        /// Whether to enable MSTX.
        #[clap(long, action)]
        msprof_tx: bool,
        /// GC detect threshold.
        #[clap(long)]
        gc_detect_threshold: Option<f32>,
        /// Whether to streamline data after analyse is complete.
        #[clap(long, value_parser = ["true", "false"], default_value = "true")]
        data_simplification: String,
        /// Types of data exported by the profiler.
        #[clap(long, value_parser = ["Text", "Db"], default_value = "Text")]
        export_type: String,
        /// Obtain the system data on the host side.
        #[clap(long, value_parser = parse_host_sys, default_value = "None")]
        host_sys: String,
        /// Whether to enable sys io.
        #[clap(long, action)]
        sys_io: bool,
        /// Whether to enable sys interconnection.
        #[clap(long, action)]
        sys_interconnection: bool,
        /// The domain that needs to be enabled in mstx mode.
        #[clap(long)]
        mstx_domain_include: Option<String>,
        /// Domains that do not need to be enabled in mstx mode.
        #[clap(long)]
        mstx_domain_exclude: Option<String>,
    },
    /// Ascend MSPTI Monitor
    NpuMonitor {
        /// Start NPU monitor.
        #[clap(long, action)]
        npu_monitor_start: bool,
        /// Stop NPU monitor.
        #[clap(long, action)]
        npu_monitor_stop: bool,
        /// NPU monitor report interval in seconds.
        #[clap(long, default_value_t = 60)]
        report_interval_s: u32,
        /// MSPTI collect activity kind
        #[clap(long, value_parser = parse_mspti_activity_kinds, default_value = "Marker")]
        mspti_activity_kind: String,
    },
    /// Pause dcgm profiling. This enables running tools like Nsight compute and avoids conflicts.
    DcgmPause {
        /// Duration to pause dcgm profiling in seconds
        #[clap(long, default_value_t = 300)]
        duration_s: i32,
    },
    /// Resume dcgm profiling
    DcgmResume,
}

struct ClientConfigPath {
    cert_path: PathBuf,
    key_path: PathBuf,
    ca_cert_path: PathBuf,
}

fn create_dyno_client(
    host: &str, 
    port: u16,
    config: &ClientConfigPath
) -> Result<StreamOwned<ClientConnection, TcpStream>> {
    let addr = (host, port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| io::Error::new(
            io::ErrorKind::NotFound,
            "Could not resolve the host address"
        ))?;

    let stream = TcpStream::connect(addr)?;

    println!("Loading CA cert from: {}", config.ca_cert_path.display());
    let mut root_store = RootCertStore::empty();
    let ca_file = File::open(&config.ca_cert_path)?;
    let mut ca_reader = BufReader::new(ca_file);
    let ca_certs = rustls_pemfile::certs(&mut ca_reader)?;
    for ca_cert in ca_certs {
        root_store.add(&Certificate(ca_cert))?;
    }

    println!("Loading client cert from: {}", config.cert_path.display());
    let cert_file = File::open(&config.cert_path)?;
    let mut cert_reader = BufReader::new(cert_file);
    let certs = rustls_pemfile::certs(&mut cert_reader)?
        .into_iter()
        .map(Certificate)
        .collect();

    println!("Loading client key from: {}", config.key_path.display());
    let key_file = File::open(&config.key_path)?;
    let mut key_reader = BufReader::new(key_file);
    let keys = rustls_pemfile::pkcs8_private_keys(&mut key_reader)?;
    if keys.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "No private key found in the key file"
        ).into());
    }
    let key = PrivateKey(keys[0].clone());

    let config = rustls::ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(root_store)
        .with_client_auth_cert(certs, key)?;

    let server_name = rustls::ServerName::try_from(host)
        .map_err(|e| io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Invalid hostname: {}", e)
        ))?;

    let conn = rustls::ClientConnection::new(
        Arc::new(config),
        server_name
    )?;

    // 返回 TLS stream
    Ok(StreamOwned::new(conn, stream))
}

fn main() -> Result<()> {
    let Opts {
        hostname,
        port,
        certs_dir,
        cmd,
    } = Opts::parse();

    let certs_dir = PathBuf::from(&certs_dir);

    let config = ClientConfigPath {
        cert_path: certs_dir.join("client.crt"),
        key_path: certs_dir.join("client.key"),
        ca_cert_path: certs_dir.join("ca.crt"),
    };

    let client = create_dyno_client(&hostname, port, &config)
        .expect("Couldn't connect to the server...");

    match cmd {
        Command::Status => status::run_status(client),
        Command::Version => version::run_version(client),
        Command::Gputrace {
            job_id,
            pids,
            log_file,
            duration_ms,
            iterations,
            profile_start_time,
            profile_start_iteration_roundup,
            process_limit,
            record_shapes,
            profile_memory,
            with_stacks,
            with_flops,
            with_modules,
        } => {
            let trigger_config = if iterations > 0 {
                GpuTraceTriggerConfig::IterationBased {
                    profile_start_iteration_roundup,
                    iterations,
                }
            } else {
                GpuTraceTriggerConfig::DurationBased {
                    profile_start_time,
                    duration_ms,
                }
            };
            let trace_options = GpuTraceOptions {
                record_shapes,
                profile_memory,
                with_stacks,
                with_flops,
                with_modules,
            };
            let trace_config = GpuTraceConfig {
                log_file,
                trigger_config,
                trace_options,
            };
            gputrace::run_gputrace(client, job_id, &pids, process_limit, trace_config)
        }
        Command::Nputrace {
            job_id,
            pids,
            log_file,
            duration_ms,
            iterations,
            profile_start_time,
            start_step,
            process_limit,
            record_shapes,
            profile_memory,
            with_stack,
            with_flops,
            with_modules,
            activities,
            analyse,
            profiler_level,
            aic_metrics,
            l2_cache,
            op_attr,
            msprof_tx,
            gc_detect_threshold,
            data_simplification,
            export_type,
            host_sys,
            sys_io,
            sys_interconnection,
            mstx_domain_include,
            mstx_domain_exclude,
        } => {
            let trigger_config = if iterations > 0 {
                NpuTraceTriggerConfig::IterationBased {
                    start_step,
                    iterations,
                }
            } else {
                NpuTraceTriggerConfig::DurationBased {
                    profile_start_time,
                    duration_ms,
                }
            };

            let trace_options = NpuTraceOptions {
                record_shapes,
                profile_memory,
                with_stack,
                with_flops,
                with_modules,
                activities,
                analyse,
                profiler_level,
                aic_metrics,
                l2_cache,
                op_attr,
                msprof_tx,
                gc_detect_threshold,
                data_simplification,
                export_type,
                host_sys,
                sys_io,
                sys_interconnection,
                mstx_domain_include,
                mstx_domain_exclude,
            };
            let trace_config = NpuTraceConfig {
                log_file,
                trigger_config,
                trace_options,
            };
            nputrace::run_nputrace(client, job_id, &pids, process_limit, trace_config)
        }
        Command::NpuMonitor {
            npu_monitor_start,
            npu_monitor_stop,
            report_interval_s,
            mspti_activity_kind,
        } => {
            let npu_mon_config = NpuMonitorConfig {
                npu_monitor_start,
                npu_monitor_stop,
                report_interval_s,
                mspti_activity_kind
            };
            npumonitor::run_npumonitor(client, npu_mon_config)
        }
        Command::DcgmPause { duration_s } => dcgm::run_dcgm_pause(client, duration_s),
        Command::DcgmResume => dcgm::run_dcgm_resume(client),
        // ... add new commands here
    }
}