#include <fstream>

#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "base/ErrorInfosManager.h"

using namespace MindStudioDebugger;

namespace MsProbeTest {

TEST(ErrorInfoTest, TestLog)
{
    std::string testDir = "./testdir";
    ASSERT_EQ(mkdir(testDir.c_str(), 0750), 0);
    ErrorInfosManager::SetLogPath(testDir + "/logfile1.log");
    LOG_CRITICAL(DebuggerErrno::ERROR_DIR_NOT_EXISTS, "Critical log content.");
    std::ifstream ifs1(testDir + "/logfile1.log", std::ios::in);
    ASSERT_TRUE(ifs1.is_open());
    std::string content1((std::istreambuf_iterator<char>(ifs1)), std::istreambuf_iterator<char>());
    ifs1.close();
    EXPECT_EQ(content1, "[CRITICAL][DIR_NOT_EXISTS]Critical log content.\n");
    LOG_ERROR(DebuggerErrno::ERROR_INVALID_OPERATION, "Error log content.");
    ifs1.open(testDir + "/logfile1.log");
    ASSERT_TRUE(ifs1.is_open());
    std::string content2((std::istreambuf_iterator<char>(ifs1)), std::istreambuf_iterator<char>());
    EXPECT_EQ(content2,
              "[CRITICAL][DIR_NOT_EXISTS]Critical log content.\n[ERROR][INVALID_OPERATION]Error log content.\n");

    ErrorInfosManager::SetLogPath(testDir + "/logfile2.log");
    LOG_WARNING(DebuggerErrno::ERROR_SYSCALL_FAILED, "Warning log content.");
    std::ifstream ifs2(testDir + "/logfile2.log", std::ios::in);
    ASSERT_TRUE(ifs2.is_open());
    std::string content3((std::istreambuf_iterator<char>(ifs2)), std::istreambuf_iterator<char>());
    ifs2.close();
    EXPECT_EQ(content3, "[WARNING][SYSCALL_FAILED]Warning log content.\n");

    ErrorInfosManager::SetLogPath(testDir + "/logfile3.log");
    LOG_INFO("Info log content.");
    LOG_DEBUG("Debug log content.");
    std::ifstream ifs3(testDir + "/logfile3.log", std::ios::in);
    ASSERT_TRUE(ifs3.is_open());
    std::string content4((std::istreambuf_iterator<char>(ifs3)), std::istreambuf_iterator<char>());
    ifs3.close();
    EXPECT_EQ(content4, "[INFO]Info log content.\n");
    TEST_ExecShellCommand("rm -rf " + testDir);

    ErrorInfosManager::SetLogPath("");
}

}
