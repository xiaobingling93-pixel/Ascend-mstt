#include <cstdlib>
#include <gtest/gtest.h>

#include "include/test_utils.hpp"
#include "base/DebuggerConfig.h"
#include "base/Environment.h"

using namespace MindStudioDebugger;
using namespace MindStudioDebugger::Environment;

namespace MsProbeTest {

TEST(EnvironmentTest, TestRankId) {
    DebuggerConfig::GetInstance().Reset();
    EXPECT_EQ(GetRankID(), -1);
    DebuggerConfig::GetInstance().LoadConfig("MindSpore", CONFIG_EXAMPLE);
    EXPECT_EQ(GetRankID(), -1);
    setenv("RANK_ID", "xxxx", 1);
    EXPECT_EQ(GetRankID(), -1);
    setenv("RANK_ID", "-5", 1);
    EXPECT_EQ(GetRankID(), -1);
    setenv("RANK_ID", "2", 1);
    EXPECT_EQ(GetRankID(), 2);

    DebuggerConfig::GetInstance().Reset();
}

}
