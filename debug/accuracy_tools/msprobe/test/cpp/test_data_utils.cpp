#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <endian.h>
#include "utils/DataUtils.h"

using namespace MindStudioDebugger;
using namespace MindStudioDebugger::DataUtils;

namespace MsProbeTest {

TEST(DataUtilsTest, TestUnpackUint64Value) {
    uint64_t dataLe = 0x0102030405060708;
    uint64_t result = UnpackUint64ValueLe(&dataLe);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    EXPECT_EQ(result, 0x0102030405060708);
#else
    EXPECT_EQ(result, 0x0807060504030201);
#endif
    uint64_t dataBe = 0x0102030405060708;
    result = UnpackUint64ValueBe(&dataBe);
#if __BYTE_ORDER == __LITTLE_ENDIAN
    EXPECT_EQ(result, 0x0807060504030201);
#else
    EXPECT_EQ(result, 0x0102030405060708);
#endif
}

TEST(DataUtilsTest, TestDataTrans) {
    size_t value = 123456;
    int64_t result = SizeToS64(value);
    EXPECT_EQ(result, 123456);
    bool exception = false;
    try {
        int64_t result = SizeToS64(static_cast<size_t>(INT64_MAX) + 1ULL);
    } catch (const std::runtime_error& e) {
        exception = true;
    }
    EXPECT_TRUE(exception);
    uint64_t num = 0x123456789ABCDEF0;
    std::string s = U64ToHexString(num);
    EXPECT_EQ(s, "0x123456789ABCDEF0");
}

TEST(DataUtilsTest, TestBFloat16) {
    float fp32 = 3.14f;
    BFloat16 bf16(fp32);
#define BF16_EQ(a, b) (-0.01f < static_cast<float>((a) - (b)) && static_cast<float>((a) - (b)) < 0.01f)
    EXPECT_TRUE(BF16_EQ(fp32, static_cast<float>(bf16)));
    EXPECT_TRUE(BF16_EQ(fp32 + fp32, static_cast<float>(bf16 + bf16)));
    EXPECT_TRUE(BF16_EQ(fp32 + fp32, bf16 + fp32));
    EXPECT_TRUE(BF16_EQ(fp32 + fp32, bf16 + fp32));
#undef BF16_EQ
}

TEST(DataUtilsTest, TestDType) {
    EXPECT_EQ(SizeOfDType(DataType::DT_FLOAT), 4);
    EXPECT_EQ(SizeOfDType(DataType::DT_DOUBLE), 8);
    EXPECT_EQ(SizeOfDType(DataType::DT_INT64), 8);
    EXPECT_EQ(SizeOfDType(DataType::DT_UINT8), 1);
    EXPECT_EQ(SizeOfDType(DataType::DT_FLOAT16), 2);
    EXPECT_EQ(SizeOfDType(static_cast<DataType>(99)), 0);
    EXPECT_EQ(GetDTypeString(DataType::DT_BOOL), "BOOL");
    EXPECT_EQ(GetDTypeString(DataType::DT_INT8), "INT8");
    EXPECT_EQ(GetDTypeString(DataType::DT_BF16), "BF16");
    EXPECT_EQ(GetDTypeString(DataType::DT_UINT64), "UINT64");
    EXPECT_EQ(GetDTypeString(DataType::DT_COMPLEX64), "COMPLEX64");
    EXPECT_EQ(GetDTypeString(static_cast<DataType>(99)), "UNKNOWN");
}

TEST(DataUtilsTest, TestGetFormatString) {
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_NCHW), "NCHW");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_NHWC), "NHWC");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_FRACTAL_Z), "FRACTAL_Z");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_C1HWNC0), "C1HWNC0");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_HWCN), "HWCN");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_C1HWNCOC0), "C1HWNCoC0");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_DHWNC), "DHWNC");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_NCL), "NCL");
    EXPECT_EQ(GetFormatString(TensorFormat::FORMAT_MAX), "UNKNOWN");
}

TEST(DataUtilsTest, GetShapeString) {
    EXPECT_EQ(GetShapeString({2, 3, 5}), "(2,3,5)");
    EXPECT_EQ(GetShapeString({}), "()");
    EXPECT_EQ(GetShapeString({3}), "(3)");
}

}
