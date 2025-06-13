#include "DynologTensorBoardLogger.h"

#include <string>

#include "hbt/src/common/System.h"

#include <fmt/format.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>
#include <chrono>
#include <glog/logging.h>

#ifdef USE_TENSORBOARD
DEFINE_string(metric_log_dir, "", "The Path to store tensorboard logs");

namespace dynolog {

const std::string TensorBoardLoggerImpl::log_file_name_ = "tfevents.pb";
std::filesystem::path TensorBoardLoggerManager::log_path_ = "";

DynologTensorBoardLogger::DynologTensorBoardLogger(const std::string& metric_log_dir)
  : logPath_(metric_log_dir) {

    if (!validateLogDir(logPath_)) {
      std::runtime_error("Unable to record logs in the target folder");
    }
  
    // logger = std::make_unique<TensorBoardLogger>(FLAGS_metric_log_dir);
    LOG(INFO) << "Initialized tensorboard logger on = "
              << FLAGS_metric_log_dir;
}

void DynologTensorBoardLogger::finalize() {
  TensorBoardLoggerManager::logPath(logPath_);
  auto logging_guard = TensorBoardLoggerManager::singleton();
  auto prom = logging_guard.manager;
  auto deviceId = dynamic_metrics_["deviceId"] == "-1" ? "host": dynamic_metrics_["deviceId"];
  auto kind = dynamic_metrics_["kind"];
  std::string real_tag = kind == "Marker" 
    ? kind + "/" + dynamic_metrics_["domain"] 
    : kind;
  std::string metric_name = "duration";
  MsptiMetricDesc desc{deviceId, kind, real_tag, metric_name, kvs_["duration"]};
  prom->log(desc);
}

bool DynologTensorBoardLogger::validateLogDir(const std::string& path) {
    std::filesystem::path log_path(path);

    if (!std::filesystem::exists(log_path)) {
        LOG(ERROR) << "Error: Path does not exist: " << path;
        return false;
    }

    if (!std::filesystem::is_directory(log_path)) {
        LOG(ERROR) << "Error: Path is not a directory: " << path;
        return false;
    }

    if (std::filesystem::is_symlink(log_path)) {
        LOG(ERROR) << "Error: Path is a symbolic link: " << path;
        return false;
    }

    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        LOG(ERROR) << "Error: Cannot stat path: " << path;
        return false;
    }

    uid_t current_uid = getuid();
    if (info.st_uid != current_uid && current_uid != 0) {
        LOG(ERROR) << "Error: Path is not owned by current user";
        return false;
    }
    return true;
}

// static
std::shared_ptr<TensorBoardLoggerManager> TensorBoardLoggerManager::singleton_() {
  static std::shared_ptr<TensorBoardLoggerManager> manager_ =
      std::make_shared<TensorBoardLoggerManager>();
  return manager_;
}

// static
TensorBoardLoggerManager::LoggingGuard TensorBoardLoggerManager::singleton() {
  auto s = singleton_();
  return LoggingGuard{.manager = s, .lock_guard = s->lock()};
}

bool TensorBoardLoggerManager::isValidMetric(const MsptiMetricDesc &desc)
{   
    auto it = validMetrics_.find(desc.kind_);
    if (it == validMetrics_.end() || !it->second.count(desc.metric_name_)) {
        return false;
    }
    return true;
}

uint64_t TensorBoardLoggerManager::getCurStep(const std::string& device, const std::string& kind)
{
  auto key = std::make_pair(device, kind);
  return device_kind2_step_[key]++;
}

void TensorBoardLoggerManager::log(const MsptiMetricDesc& desc) {
  if (!isValidMetric(desc)) {
    return;
  }

  auto device = desc.device_id_;
  // 读取tensorboardImpl，调用Log方法写入
  auto it = device_loggers_.find(device);
  std::shared_ptr<TensorBoardLoggerImpl> logger;
  if (it == device_loggers_.end()) {
    std::string device_log_path = log_path_ / ("device_" + device);
    device_loggers_[device] = std::make_shared<TensorBoardLoggerImpl>(device_log_path, "");
  }
  logger = device_loggers_[device];
  logger->log(desc.tag_, desc.val_, getCurStep(device, desc.kind_));
}

void TensorBoardLoggerImpl::log(const std::string &key, double val, uint64_t step) {
    if (!std::filesystem::exists(log_path_)) {
        std::error_code ec;
        std::filesystem::create_directories(log_path_, ec);
        if (ec) {
            LOG(ERROR) << "failed to create log dir: " << log_path_ << "errorcode: " << ec.message();
            return;
        }
    }

    if (logger_ == nullptr) {
      logger_ = std::make_shared<TensorBoardLogger>(log_path_ / log_file_name_);
    }
    logger_->add_scalar(key, step, val);
}
}
#endif