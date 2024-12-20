#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include "include/test_utils.hpp"
#include "third_party/ACL/AclApi.hpp"
#include "base/ErrorInfos.hpp"
#include "core/PrecisionDebugger.hpp"

using namespace MindStudioDebugger;

namespace MsProbeTest {

class PrecisionDbgTaskStub : public PrecisionDbgTaskBase {
public:
    PrecisionDbgTaskStub() = default;
    ~PrecisionDbgTaskStub() = default;
    std::string Name() const override {return "PrecisionDbgTaskStub";}
    bool Condition(const DebuggerConfig& cfg) const override {return true;}

    void Initialize(const DebuggerConfig& cfg) {initialize_called = true;}
    void OnStart() {start_called = true;}
    void OnStop() {stop_called = true;}
    void OnStep() {step_called = true;}

    bool initialize_called{false};
    bool start_called{false};
    bool stop_called{false};
    bool step_called{false};
};

class PrecisionDbgTaskUselessStub : public PrecisionDbgTaskStub {
public:
    bool Condition(const DebuggerConfig& cfg) const override {return false;}
};

TEST(PrecisionDebuggerTest, TestRegisterBeforeInit) {
    PrecisionDebugger& debugger = PrecisionDebugger::GetInstance();
    PrecisionDbgTaskStub stub_task;

    DebuggerConfig::GetInstance().Reset();
    debugger.RegisterDebuggerTask(&stub_task);
    stub_task.Register();

    EXPECT_FALSE(debugger.IsEnable());
    EXPECT_EQ(debugger.GetCurStep(), 0);
    debugger.Start();
    EXPECT_FALSE(debugger.IsEnable());
    debugger.Stop();
    debugger.Step();
    EXPECT_EQ(debugger.GetCurStep(), 0);

    EXPECT_FALSE(stub_task.initialize_called);
    EXPECT_FALSE(stub_task.start_called);
    EXPECT_FALSE(stub_task.stop_called);
    EXPECT_FALSE(stub_task.step_called);

    debugger.UnRegisterDebuggerTask(&stub_task);
    debugger.UnRegisterDebuggerTask(nullptr);
}

TEST(PrecisionDebuggerTest, TestInit) {
    PrecisionDebugger& debugger = PrecisionDebugger::GetInstance();
    MOCKER(MindStudioDebugger::AscendCLApi::LoadAclApi)
    .stubs()
    .then(returnValue(0))
    .expects(atLeast(1));

    DebuggerConfig::GetInstance().Reset();
    EXPECT_FALSE(debugger.HasInitialized());
    EXPECT_NE(debugger.Initialize("", ""), 0);
    EXPECT_FALSE(debugger.HasInitialized());
    CleanErrorInfoCache();
    EXPECT_EQ(debugger.Initialize("MindSpore", CONFIG_EXAMPLE), 0);
    EXPECT_TRUE(debugger.HasInitialized());
    EXPECT_EQ(debugger.Initialize("MindSpore", CONFIG_EXAMPLE), 0);
    EXPECT_TRUE(debugger.HasInitialized());

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

TEST(PrecisionDebuggerTest, TestSubTaskDispatch) {
    PrecisionDebugger& debugger = PrecisionDebugger::GetInstance();
    PrecisionDbgTaskStub stub_task1;
    PrecisionDbgTaskStub stub_task2;
    PrecisionDbgTaskUselessStub stub_task3;
    MOCKER(MindStudioDebugger::AscendCLApi::LoadAclApi)
    .stubs()
    .then(returnValue(0));
    MOCKER(MindStudioDebugger::AscendCLApi::ACLAPI_aclrtSynchronizeDevice)
    .stubs()
    .then(returnValue(0))
    .expects(atLeast(1));

    stub_task1.Register();
    EXPECT_EQ(debugger.Initialize("MindSpore", CONFIG_EXAMPLE), 0);
    stub_task2.Register();
    stub_task3.Register();

    EXPECT_TRUE(stub_task1.initialize_called);
    EXPECT_TRUE(stub_task2.initialize_called);
    EXPECT_FALSE(stub_task3.initialize_called);
    EXPECT_FALSE(stub_task1.start_called);
    EXPECT_FALSE(stub_task2.stop_called);
    EXPECT_FALSE(stub_task3.step_called);

    debugger.Start();
    EXPECT_TRUE(stub_task1.start_called);
    EXPECT_FALSE(stub_task3.start_called);

    debugger.Stop();
    EXPECT_TRUE(stub_task1.stop_called);
    EXPECT_TRUE(stub_task2.stop_called);

    debugger.Step();
    EXPECT_TRUE(stub_task1.step_called);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

}
