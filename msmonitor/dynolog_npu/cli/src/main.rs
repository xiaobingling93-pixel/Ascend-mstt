// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use std::fs::File;
use std::io::{BufReader, Read};
use rustls::{Certificate, RootCertStore, PrivateKey, ClientConnection, StreamOwned};
use std::sync::Arc;
use std::net::TcpStream;
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::io;
use rpassword::prompt_password;

use anyhow::Result;
use clap::Parser;
use std::collections::HashSet;

use x509_parser::prelude::*;
use x509_parser::num_bigint::ToBigInt;
use std::fs::read_to_string;
use x509_parser::public_key::RSAPublicKey;
use x509_parser::der_parser::oid;
use num_bigint::BigUint;
use openssl::pkey::PKey;

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
const MIN_RSA_KEY_LENGTH: u64 = 3072; // 最小 RSA 密钥长度（位）

#[derive(Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Opts {
    #[arg(long, default_value = "localhost")]
    hostname: String,
    #[arg(long, default_value_t = DYNO_PORT)]
    port: u16,
    #[arg(long, required = true)]
    certs_dir: String,
    #[command(subcommand)]
    cmd: Command,
}

const ALLOWED_VALUES: &[&str] = &["Marker", "Kernel", "API", "Hccl", "Memory", "MemSet", "MemCpy", "Communication"];

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
        #[arg(long, default_value_t = 0)]
        job_id: u64,
        /// List of pids to capture trace for (comma separated).
        #[arg(long, default_value = "0")]
        pids: String,
        /// Duration of trace to collect in ms.
        #[arg(long, default_value_t = 500)]
        duration_ms: u64,
        /// Training iterations to collect, this takes precedence over duration.
        #[arg(long, default_value_t = -1)]
        iterations: i64,
        /// Log file for trace.
        #[arg(long)]
        log_file: String,
        /// Unix timestamp used for synchronized collection (milliseconds since epoch).
        #[arg(long, default_value_t = 0)]
        profile_start_time: u64,
        /// Start iteration roundup, starts an iteration based trace at a multiple
        /// of this value.
        #[arg(long, default_value_t = 1)]
        profile_start_iteration_roundup: u64,
        /// Max number of processes to profile.
        #[arg(long, default_value_t = 3)]
        process_limit: u32,
        /// Record PyTorch operator input shapes and types.
        #[arg(long)]
        record_shapes: bool,
        /// Profile PyTorch memory.
        #[arg(long)]
        profile_memory: bool,
        /// Capture Python stacks in traces.
        #[arg(long)]
        with_stacks: bool,
        /// Annotate operators with analytical flops.
        #[arg(long)]
        with_flops: bool,
        /// Capture PyTorch operator modules in traces.
        #[arg(long)]
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

fn verify_certificate(cert_der: &[u8], is_root_cert: bool) -> Result<()> {
    // 解析 X509 证书
    let (_, cert) = X509Certificate::from_der(cert_der)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Failed to parse cert: {:?}", e)))?;

    // 检查证书版本是否为 X.509v3
    if cert.tbs_certificate.version != X509Version(2) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Certificate is not X.509v3"
        ).into());
    }

    // 检查证书签名算法
    let sig_alg = cert.signature_algorithm.algorithm;
    
    // 定义不安全的算法 OID
    let md2_rsa = oid!(1.2.840.113549.1.1.2);  // MD2 with RSA
    let md5_rsa = oid!(1.2.840.113549.1.1.4);  // MD5 with RSA
    let sha1_rsa = oid!(1.2.840.113549.1.1.5); // SHA1 with RSA
    
    // 检查是否使用不安全的算法
    if sig_alg == md2_rsa || sig_alg == md5_rsa || sig_alg == sha1_rsa {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Certificate uses insecure signature algorithm"
        ).into());
    }

    // 定义 RSA 签名算法 OID
    let rsa_sha256 = oid!(1.2.840.113549.1.1.11); // RSA with SHA256
    let rsa_sha384 = oid!(1.2.840.113549.1.1.12); // RSA with SHA384
    let rsa_sha512 = oid!(1.2.840.113549.1.1.13); // RSA with SHA512

    // 检查 RSA 密钥长度
    if sig_alg == rsa_sha256 || sig_alg == rsa_sha384 || sig_alg == rsa_sha512 {
        // 获取公钥
        if let Ok((_, public_key)) = SubjectPublicKeyInfo::from_der(&cert.tbs_certificate.subject_pki.subject_public_key.data) {
            if let Ok((_, rsa_key)) = RSAPublicKey::from_der(&public_key.subject_public_key.data) {
                // 检查 RSA 密钥长度
                let modulus = BigUint::from_bytes_be(&rsa_key.modulus);
                let key_length = modulus.bits();
                if key_length < MIN_RSA_KEY_LENGTH {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("RSA key length {} bits is less than required {} bits", key_length, MIN_RSA_KEY_LENGTH)
                    ).into());
                }
            }
        }
    }

    // 检查证书的扩展域
    let mut has_ca_constraint = false;
    let mut has_key_usage = false;
    let mut has_crl_sign = false;
    let mut has_cert_sign = false;

    for ext in cert.tbs_certificate.extensions() {
        if ext.oid == oid_registry::OID_X509_EXT_BASIC_CONSTRAINTS {
            if let Ok((_, constraints)) = BasicConstraints::from_der(ext.value) {
                has_ca_constraint = constraints.ca;
            } else {
                println!("Failed to parse Basic Constraints");
            }
        } else if ext.oid == oid_registry::OID_X509_EXT_KEY_USAGE {
            println!("Found Key Usage extension");
            if let Ok((_, usage)) = KeyUsage::from_der(ext.value) {
                has_key_usage = true;
                has_cert_sign = usage.key_cert_sign();
                has_crl_sign = usage.crl_sign();
            } else {
                println!("Failed to parse Key Usage");
            }
        }
    }

    // 根据证书类型进行不同的验证
    if is_root_cert {
        // 根证书验证要求
        if !has_ca_constraint {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Root certificate must have CA constraint"
            ).into());
        }
        if !has_key_usage {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Root certificate must have key usage extension"
            ).into());
        }
        if !has_cert_sign {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Root certificate must have certificate signature permission"
            ).into());
        }
        if !has_crl_sign {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Root certificate must have CRL signature permission"
            ).into());
        }
    } else {
        // 客户端证书验证要求
        if has_ca_constraint {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Client certificate should not have CA constraint"
            ).into());
        }
        if !has_key_usage {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Client certificate must have key usage extension"
            ).into());
        }
    }

    // 检查证书有效期
    let now = chrono::Utc::now();
    let not_before = chrono::DateTime::from_timestamp(
        cert.tbs_certificate.validity.not_before.timestamp(),
        0
    ).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid not_before date"))?;

    let not_after = chrono::DateTime::from_timestamp(
        cert.tbs_certificate.validity.not_after.timestamp(),
        0
    ).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid not_after date"))?;

    if now < not_before {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Certificate is not yet valid. Valid from: {}", not_before)
        ).into());
    }

    if now > not_after {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Certificate has expired. Expired at: {}", not_after)
        ).into());
    }

    Ok(())
}

fn is_cert_revoked(cert_der: &[u8], crl_path: &PathBuf) -> Result<bool> {
    // 解析 X509 证书
    let (_, cert) = X509Certificate::from_der(cert_der)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Failed to parse cert: {:?}", e)))?;

    // 读取 CRL 文件
    let crl_data = read_to_string(crl_path)?;
    let (_, pem) = pem::parse_x509_pem(crl_data.as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Failed to parse CRL PEM: {:?}", e)))?;
    
    // 解析 CRL
    let (_, crl) = CertificateRevocationList::from_der(&pem.contents)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Failed to parse CRL: {:?}", e)))?;

    // 检查 CRL 的有效期
    let now = chrono::Utc::now();
    let crl_not_before = chrono::DateTime::from_timestamp(
        crl.tbs_cert_list.this_update.timestamp(),
        0
    ).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid CRL this_update date"))?;

    let crl_not_after = if let Some(next_update) = crl.tbs_cert_list.next_update {
        chrono::DateTime::from_timestamp(
            next_update.timestamp(),
            0
        ).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid CRL next_update date"))?
    } else {
        crl_not_before + chrono::Duration::days(365)
    };

    // 检查 CRL 是否在有效期内
    if now < crl_not_before {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("CRL is not yet valid. Valid from: {}", crl_not_before)
        ).into());
    }

    if now > crl_not_after {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("CRL has expired. Expired at: {}", crl_not_after)
        ).into());
    }

    // 获取证书序列号
    let cert_serial = cert.tbs_certificate.serial.to_bigint()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Failed to convert certificate serial to BigInt"))?;

    // 检查 CRL 吊销条目
    for revoked in crl.iter_revoked_certificates() {
        let revoked_serial = revoked.user_certificate.to_bigint()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Failed to convert revoked certificate serial to BigInt"))?;
        
        if revoked_serial == cert_serial {
            return Ok(true);
        }
    }
    Ok(false)
}

enum DynoClient {
    Secure(StreamOwned<ClientConnection, TcpStream>),
    Insecure(TcpStream),
}

fn create_dyno_client(
    host: &str, 
    port: u16,
    certs_dir: &str,
) -> Result<DynoClient> {
    if certs_dir == "NO_CERTS" {
        println!("Running in no-certificate mode");
        create_dyno_client_with_no_certs(host, port)
    } else {
        println!("Running in certificate mode");
        let certs_dir = PathBuf::from(certs_dir);
        let config = ClientConfigPath {
            cert_path: certs_dir.join("client.crt"),
            key_path: certs_dir.join("client.key"),
            ca_cert_path: certs_dir.join("ca.crt"),
        };
        let client = create_dyno_client_with_certs(host, port, &config)?;
        Ok(DynoClient::Secure(client))
    }
}

fn create_dyno_client_with_no_certs(
    host: &str, 
    port: u16,
) -> Result<DynoClient> {
    let addr = (host, port)
        .to_socket_addrs()?
        .next()
        .expect("Failed to connect to the server");
    let stream = TcpStream::connect(addr)?;
    Ok(DynoClient::Insecure(stream))
}

// 安全清除密码的函数
fn secure_clear_password(password: &mut Vec<u8>) {
    if !password.is_empty() {
        // 使用零覆盖密码数据
        for byte in password.iter_mut() {
            *byte = 0;
        }
        // 清空向量
        password.clear();
        // 收缩向量容量，释放内存
        password.shrink_to_fit();
    }
}

fn create_dyno_client_with_certs(
    host: &str, 
    port: u16,
    config: &ClientConfigPath,
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
    for ca_cert in &ca_certs {
        verify_certificate(ca_cert, true)?;  // 验证根证书
    }
    for ca_cert in ca_certs {
        root_store.add(&Certificate(ca_cert))?;
    }

    println!("Loading client cert from: {}", config.cert_path.display());
    let cert_file = File::open(&config.cert_path)?;
    let mut cert_reader = BufReader::new(cert_file);
    let certs = rustls_pemfile::certs(&mut cert_reader)?;
    
    // 检查客户端证书的基本要求
    for cert in &certs {
        verify_certificate(cert, false)?;  // 验证客户端证书
    }

    // 检查证书吊销状态
    let crl_path = config.cert_path.parent().unwrap().join("ca.crl");
    if crl_path.exists() {
        println!("Checking CRL file: {}", crl_path.display());
        for cert in &certs {
            match is_cert_revoked(cert, &crl_path) {
                Ok(true) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Certificate is revoked"
                    ).into());
                }
                Ok(false) => {
                    continue;
                }
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("CRL verification failed: {}", e)
                    ).into());
                }
            }
        }
    } else {
        println!("CRL file does not exist: {}", crl_path.display());
    }

    let certs = certs.into_iter().map(Certificate).collect();

    println!("Loading client key from: {}", config.key_path.display());
    let key_file = File::open(&config.key_path)?;
    let mut key_reader = BufReader::new(key_file);
    
    // 检查私钥是否加密
    let mut key_data = Vec::new();
    key_reader.read_to_end(&mut key_data)?;
    let key_str = String::from_utf8_lossy(&key_data);
    let is_encrypted = key_str.contains("ENCRYPTED");

    // 根据是否加密来加载私钥
    let keys = if is_encrypted {
        // 如果私钥是加密的，请求用户输入密码
        let mut password = prompt_password("Please enter the certificate password: ")?.into_bytes();
        let pkey = PKey::private_key_from_pem_passphrase(&key_data, &password)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Failed to decrypt private key: {}", e)))?;
        
        // 手动清除密码
        secure_clear_password(&mut password);
        
        // 返回私钥
        vec![pkey.private_key_to_der()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Failed to convert private key to DER: {}", e)))?]
    } else {
        // 如果私钥未加密，直接加载
        let mut key_reader = BufReader::new(File::open(&config.key_path)?);
        rustls_pemfile::pkcs8_private_keys(&mut key_reader)?
    };
    
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

    Ok(StreamOwned::new(conn, stream))
}


fn main() -> Result<()> {
    let Opts {
        hostname,
        port,
        certs_dir,
        cmd,
    } = Opts::parse();

    let client = create_dyno_client(&hostname, port, &certs_dir)
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