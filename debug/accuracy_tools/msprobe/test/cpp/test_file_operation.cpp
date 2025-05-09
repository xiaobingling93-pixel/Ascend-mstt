#include <gtest/gtest.h>
#include <fstream>
#include <string>
#include <vector>

#include "test_utils.hpp"
#include "utils/DataUtils.h"
#include "utils/FileOperation.h"

using namespace MindStudioDebugger;
using namespace MindStudioDebugger::FileOperation;

namespace MsProbeTest {

TEST(FileOperationTest, TestDumpJson) {
    std::string testPath = "./test.json";
    nlohmann::json testJson = {{"key", "value"}};
    auto result = DumpJson(testPath, testJson);
    EXPECT_EQ(result, DebuggerErrno::OK);

    std::ifstream ifs(testPath);
    std::string fileContent((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    EXPECT_EQ(fileContent, testJson.dump());
    remove(testPath.c_str());
}

TEST(FileOperationTest, TestDumpNpy) {
    std::string testPath = "./test.npy";
    std::vector<uint8_t> int8Data = {0, 1, 2, 3, 4, 5};
    auto result = DumpNpy(testPath, int8Data.data(), int8Data.size() * sizeof(uint8_t), DataUtils::DataType::DT_UINT8,
                          {2, 3});
    EXPECT_EQ(result, DebuggerErrno::OK);
    std::string content = TEST_ExecShellCommand("python -c \'import numpy; print(numpy.load(\"./test.npy\"))\'");
    EXPECT_EQ(content, "[[0 1 2]\n [3 4 5]]\n");
    remove(testPath.c_str());

    std::vector<float> fp32Data = {0.1f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f, 6.7f, 7.8f};
    result = DumpNpy(testPath, reinterpret_cast<uint8_t*>(fp32Data.data()), fp32Data.size() * sizeof(float),
                          DataUtils::DataType::DT_FLOAT, {2, 2, 2});
    EXPECT_EQ(result, DebuggerErrno::OK);
    content = TEST_ExecShellCommand("python -c \'import numpy; print(numpy.load(\"./test.npy\"))\'");
    EXPECT_EQ(content, "[[[0.1 1.2]\n  [2.3 3.4]]\n\n [[4.5 5.6]\n  [6.7 7.8]]]\n");
    remove(testPath.c_str());
}

}
