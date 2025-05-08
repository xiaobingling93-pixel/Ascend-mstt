#include <fstream>
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "test_utils.hpp"
#include "base/ErrorInfosManager.h"
#include "base/DebuggerConfig.h"

using namespace MindStudioDebugger;

namespace MsProbeTest {

static const std::string CFG_CONTENT = R"({
    "task": "statistics",
    "dump_path": "./dump_path", 
    "rank": [],
    "step": [],
    "level": "L1",
    "seed": 1234,
    "is_deterministic": false,
    "enable_dataloader": false,
    "acl_config": "",
    "tensor": {
        "scope": [],
        "list":[],
        "data_mode": ["all"],
        "backward_input": [],
        "file_format": "npy"
    },
    "statistics": {
        "scope": [],
        "list":[],
        "data_mode": ["all"],
        "summary_mode": "statistics"
    },
    "overflow_check": {
        "overflow_nums": 1,
        "check_mode":"all"
    },
    "run_ut": {
        "white_list": [],
        "black_list": [],
        "error_data_path": "./"
    },
    "grad_probe": {
        "grad_level": "L1",
        "param_list": [],
        "bounds": [-1, 0, 1]
    },
    "free_benchmark": {
        "scope": [],
        "list": [],
        "fuzz_device": "npu",
        "pert_mode": "improve_precision",
        "handler_type": "check",
        "fuzz_level": "L1",
        "fuzz_stage": "forward",
        "if_preheat": false,
        "preheat_step": 15,
        "max_sample": 20
    }
})";

class TestConfigPyTorch : public ::testing::Test
{
protected:
    void SetUp(){}
    void TearDown(){}
};

class TestConfigMindSpore : public ::testing::Test
{
protected:
    void SetUp();
    void TearDown();
    int32_t DumpCfgFile();
    const std::string framework = "MindSpore";
    const std::string cfgPath = "./config.json";
    nlohmann::json cfgJson;
    const std::string logpath = "./test.log";
};

int32_t TestConfigMindSpore::DumpCfgFile()
{
    std::ofstream ofs(cfgPath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        return -1;
    }
    try {
        ofs << cfgJson.dump();
    } catch (std::exception &e) {
        ofs.close();
        return -1;
    }

    if (ofs.fail()) {
        return -1;
    }

    return 0;
}

void TestConfigMindSpore::SetUp()
{
    DebuggerConfig::GetInstance().Reset();
    CleanErrorInfoCache();
    ErrorInfosManager::SetLogPath(logpath);
    cfgJson = nlohmann::json::parse(CFG_CONTENT);
}

void TestConfigMindSpore::TearDown()
{
    TEST_ExecShellCommand("rm -f " + cfgPath);
    TEST_ExecShellCommand("rm -f " + logpath);
}

TEST_F(TestConfigMindSpore, TestDefaultValue)
{
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    EXPECT_FALSE(cfg.IsCfgLoaded());
    EXPECT_EQ(cfg.GetFramework(), DebuggerFramework::FRAMEWORK_PYTORCH);
    EXPECT_TRUE(cfg.GetTaskList().empty());
    EXPECT_EQ(cfg.GetOutputPath(), "./output");
    EXPECT_TRUE(cfg.GetRankRange().empty());
    EXPECT_TRUE(cfg.GetStepRange().empty());
    EXPECT_EQ(cfg.GetDebugLevel(), DebuggerLevel::L1);
    EXPECT_EQ(cfg.GetRandSeed(), 1234);
    EXPECT_FALSE(cfg.IsDeterministic());
    EXPECT_FALSE(cfg.IsDataloaderEnable());
    EXPECT_EQ(cfg.GetStatisticsCfg(), nullptr);
    EXPECT_EQ(cfg.GetDumpTensorCfg(), nullptr);
    EXPECT_EQ(cfg.GetOverflowCheckCfg(), nullptr);
}

TEST_F(TestConfigMindSpore, TestLoadConfigBase)
{
    int32_t ret;
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    ret = cfg.LoadConfig("", cfgPath);
    EXPECT_EQ(ret, -1);
    CleanErrorInfoCache();
    ret = cfg.LoadConfig(framework, "./xxx");
    EXPECT_EQ(ret, -1);
    TEST_ExecShellCommand("echo \"invalid content\" > ./invalid.json");
    CleanErrorInfoCache();
    ret = cfg.LoadConfig(framework, "./invalid.json");
    EXPECT_EQ(ret, -1);
    TEST_ExecShellCommand("rm ./invalid.json");
    ASSERT_EQ(DumpCfgFile(), 0);
    CleanErrorInfoCache();
    ret = cfg.LoadConfig(framework, cfgPath);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestConfigMindSpore, TestCommonCfg)
{
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();

    /* test static method */
    EXPECT_TRUE(cfg.IsRankHits(0));
    EXPECT_TRUE(cfg.IsRankHits(7));
    EXPECT_TRUE(cfg.IsRankHits(12345));
    EXPECT_TRUE(cfg.IsStepHits(0));
    EXPECT_TRUE(cfg.IsStepHits(7));
    EXPECT_TRUE(cfg.IsStepHits(12345));

    cfgJson["dump_path"] = "./output1";
    cfgJson["rank"] = nlohmann::json::array({0, 1, 8});
    cfgJson["step"] = nlohmann::json::array({2, 4, "6-8"});
    cfgJson["level"] = "L2";
    cfgJson["seed"] = 2345;
    cfgJson["is_deterministic"] = true;
    cfgJson["enable_dataloader"] = true;
    ASSERT_EQ(DumpCfgFile(), 0);
    EXPECT_EQ(cfg.LoadConfig(framework, cfgPath), 0);
    EXPECT_EQ(cfg.GetTaskList(), std::vector<DebuggerTaskType>({DebuggerTaskType::TASK_DUMP_STATISTICS}));
    EXPECT_EQ(cfg.GetOutputPath(), Trim(TEST_ExecShellCommand("realpath ./output1")));
    EXPECT_EQ(cfg.GetRankRange(), std::vector<uint32_t>({0, 1, 8}));
    EXPECT_EQ(cfg.GetStepRange(), std::vector<uint32_t>({2, 4, 6, 7, 8}));
    EXPECT_EQ(cfg.GetDebugLevel(), DebuggerLevel::L2);
    EXPECT_EQ(cfg.GetRandSeed(), 2345);
    EXPECT_TRUE(cfg.IsDeterministic());
    EXPECT_TRUE(cfg.IsDataloaderEnable());
    EXPECT_NE(cfg.GetStatisticsCfg(), nullptr);
    EXPECT_EQ(cfg.GetDumpTensorCfg(), nullptr);
    EXPECT_EQ(cfg.GetOverflowCheckCfg(), nullptr);
    EXPECT_TRUE(cfg.IsRankHits(0));
    EXPECT_FALSE(cfg.IsRankHits(7));
    EXPECT_FALSE(cfg.IsRankHits(12345));
    EXPECT_TRUE(cfg.IsStepHits(4));
    EXPECT_TRUE(cfg.IsStepHits(6));
    EXPECT_TRUE(cfg.IsStepHits(8));
    EXPECT_FALSE(cfg.IsStepHits(9));

    /* invalid case */
    cfg.Reset();
    ErrorInfosManager::SetLogPath("./test.log");
    cfgJson["dump_path"] = 111;
    cfgJson["rank"] = "abc";
    cfgJson["step"] = nlohmann::json::array({"a", "b"});
    cfgJson["level"] = "L10";
    cfgJson["seed"] = "123";
    cfgJson["is_deterministic"] = 1;
    cfgJson["enable_dataloader"] = "true";
    ASSERT_EQ(DumpCfgFile(), 0);
    EXPECT_NE(cfg.LoadConfig(framework, cfgPath), 0);
    std::string logContent = TEST_ExecShellCommand("cat " + logpath);
    EXPECT_NE(logContent.find("dump_path"), std::string::npos);
    EXPECT_NE(logContent.find("rank"), std::string::npos);
    EXPECT_NE(logContent.find("step"), std::string::npos);
    EXPECT_NE(logContent.find("level"), std::string::npos);
    EXPECT_NE(logContent.find("seed"), std::string::npos);
    EXPECT_NE(logContent.find("is_deterministic"), std::string::npos);
    EXPECT_NE(logContent.find("enable_dataloader"), std::string::npos);
}

TEST_F(TestConfigMindSpore, TestTensorCfg)
{
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    cfgJson["task"] = "tensor";
    cfgJson["level"] = "L2";
    nlohmann::json& tensorCfgJson = cfgJson["tensor"];
    tensorCfgJson["scope"] = nlohmann::json::array({"a", "b"});
    tensorCfgJson["list"] = nlohmann::json::array({"name-regex(conv)", "add", "ReduceMean-op0.10.5"});
    tensorCfgJson["data_mode"] = nlohmann::json::array({"all"});
    tensorCfgJson["backward_input"] = nlohmann::json::array({"/a.pt", "/b.pt"});;
    tensorCfgJson["file_format"] = "npy";
    ASSERT_EQ(DumpCfgFile(), 0);
    EXPECT_EQ(cfg.LoadConfig(framework, cfgPath), 0);
    std::shared_ptr<const DumpTensorCfg> tensorcfg = cfg.GetDumpTensorCfg();
    ASSERT_NE(tensorcfg, nullptr);
    EXPECT_EQ(tensorcfg->scope, std::vector<std::string>({"a", "b"}));
    EXPECT_EQ(tensorcfg->list, std::vector<std::string>({"name-regex(conv)", "add", "ReduceMean-op0.10.5"}));
    EXPECT_EQ(tensorcfg->direction, DebuggerDataDirection::DIRECTION_BOTH);
    EXPECT_EQ(tensorcfg->inout, DebuggerDataInOut::INOUT_BOTH);
    EXPECT_EQ(tensorcfg->backwardInput, std::vector<std::string>({"/a.pt", "/b.pt"}));
    EXPECT_EQ(tensorcfg->fileFormat, DebuggerDumpFileFormat::FILE_FORMAT_NPY);
}

TEST_F(TestConfigMindSpore, TestStatisticCfg)
{
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    cfgJson["task"] = "statistics";
    cfgJson["level"] = "L2";
    nlohmann::json& statisticsCfgJson = cfgJson["statistics"];
    statisticsCfgJson["scope"] = nlohmann::json::array({"c", "d"});
    statisticsCfgJson["list"] = nlohmann::json::array({"name-regex(conv)", "add", "ReduceMean-op0.10.5"});
    statisticsCfgJson["data_mode"] = nlohmann::json::array({"input"});
    statisticsCfgJson["summary_mode"] = "statistics";
    ASSERT_EQ(DumpCfgFile(), 0);
    EXPECT_EQ(cfg.LoadConfig(framework, cfgPath), 0);
    std::shared_ptr<const StatisticsCfg> statisticscfg = cfg.GetStatisticsCfg();
    ASSERT_NE(statisticscfg, nullptr);
    EXPECT_EQ(statisticscfg->scope, std::vector<std::string>({"c", "d"}));
    EXPECT_EQ(statisticscfg->list, std::vector<std::string>({"name-regex(conv)", "add", "ReduceMean-op0.10.5"}));
    EXPECT_EQ(statisticscfg->direction, DebuggerDataDirection::DIRECTION_BOTH);
    EXPECT_EQ(statisticscfg->inout, DebuggerDataInOut::INOUT_INPUT);
    EXPECT_EQ(statisticscfg->summaryOption,std::vector<DebuggerSummaryOption>(
        {DebuggerSummaryOption::MAX, DebuggerSummaryOption::MIN, DebuggerSummaryOption::MEAN, DebuggerSummaryOption::L2NORM}));
}

TEST_F(TestConfigMindSpore, TestOverflowCfg)
{
    DebuggerConfig& cfg = DebuggerConfig::GetInstance();
    cfgJson["task"] = "overflow_check";
    nlohmann::json& overflowCfgJson = cfgJson["overflow_check"];
    overflowCfgJson["overflow_nums"] = 3;
    overflowCfgJson["check_mode"] = "all";
    ASSERT_EQ(DumpCfgFile(), 0);
    EXPECT_EQ(cfg.LoadConfig(framework, cfgPath), 0);
    std::shared_ptr<const OverflowCheckCfg> overflowcfg = cfg.GetOverflowCheckCfg();
    ASSERT_NE(overflowcfg, nullptr);
    EXPECT_EQ(overflowcfg->overflowNums, 3);
    EXPECT_EQ(overflowcfg->checkMode, DebuggerOpCheckLevel::CHECK_LEVEL_ALL);
}

}
