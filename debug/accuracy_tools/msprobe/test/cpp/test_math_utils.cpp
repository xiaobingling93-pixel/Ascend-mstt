#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <string>
#include <sstream>
#include "utils/MathUtils.h"

using namespace MindStudioDebugger;
using namespace MindStudioDebugger::MathUtils;

namespace MsProbeTest {

TEST(MathUtilsTest, TestRandom)
{
    for (uint32_t i = 0; i < 5; i++) {
        float result = Random();
        EXPECT_GE(result, 0.0f);
        EXPECT_LE(result, 1.0f);
    }
    for (uint32_t i = 0; i < 5; i++) {
        float floor = static_cast<float>(i * 5) - 10.0f;
        float ceil = static_cast<float>(i * 10);
        float result = Random(floor, ceil);
        EXPECT_GE(result, floor);
        EXPECT_LE(result, ceil);
    }
}

TEST(MathUtilsTest, TestRandomInt)
{
    for (uint32_t i = 0; i < 5; i++) {
        int32_t floor = static_cast<int32_t>(i * 5) - 10;
        int32_t ceil = static_cast<int32_t>(i * 10);
        int32_t result = RandomInt(floor, ceil);
        EXPECT_GE(result, floor);
        EXPECT_LT(result, ceil);
    }
}

TEST(MathUtilsTest, TestRandomString)
{
    uint32_t len = 16;
    std::string result = RandomString(len);
    EXPECT_EQ(result.length(), len);
    for (char c : result) {
        EXPECT_TRUE((c >= ' ' && c <= '~'));
    }

    result = RandomString(len, 'a', 'f');
    EXPECT_EQ(result.length(), len);
    for (char c : result) {
        EXPECT_TRUE(c >= 'a' && c <= 'f');
    }
}

TEST(MathUtilsTest, TestCalculateMD5)
{
    const uint8_t data[] = "Hello, world!";
    std::string result = CalculateMD5(data, sizeof(data) - 1);
    EXPECT_EQ(result, "6cd3556deb0da54bca060b4c39479839");
}

TEST(MathUtilsTest, TestGcd)
{
    EXPECT_EQ(Gcd(10, 5), 5);
    EXPECT_EQ(Gcd(15, 5), 5);
    EXPECT_EQ(Gcd(0, 5), 0);
    EXPECT_EQ(Gcd(5, 0), 0);
    EXPECT_EQ(Gcd(0, 0), 0);
    EXPECT_EQ(Gcd(1, 1), 1);
}

TEST(MathUtilsTest, TestLcm)
{
    EXPECT_EQ(Lcm(10, 5), 10);
    EXPECT_EQ(Lcm(15, 5), 15);
    EXPECT_EQ(Lcm(0, 5), 0);
    EXPECT_EQ(Lcm(5, 0), 0);
    EXPECT_EQ(Lcm(0, 0), 0);
    EXPECT_EQ(Lcm(1, 1), 1);
}

TEST(MathUtilsTest, TestDivCeil)
{
    EXPECT_EQ(DivCeil(10, 5), 2);
    EXPECT_EQ(DivCeil(10, 3), 4);
    EXPECT_EQ(DivCeil(10, 1), 10);
    EXPECT_EQ(DivCeil(0, 5), 0);
    EXPECT_EQ(DivCeil(0, 0), 0);
}

TEST(MathUtilsTest, TestAlignCeil)
{
    EXPECT_EQ(AlignCeil(10, 5), 10);
    EXPECT_EQ(AlignCeil(7, 5), 10);
    EXPECT_EQ(AlignCeil(0, 5), 0);
    EXPECT_EQ(AlignCeil(10, 0), 0);
}

}
