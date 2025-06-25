#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include "include/test_utils.hpp"
#include "third_party/ACL/AclApi.h"
#include "base/ErrorInfosManager.h"
#include "core/PrecisionDebugger.h"

using namespace MindStudioDebugger;

namespace MsProbeTest {

class PrecisionDbgTaskStub : public PrecisionDbgTaskBase {
public:
    PrecisionDbgTaskStub() = default;
    ~PrecisionDbgTaskStub() = default;
    std::string Name() const override {return "PrecisionDbgTaskStub";}
    bool Condition(const DebuggerConfig& cfg) const override {return true;}

    void Initialize(const DebuggerConfig& cfg) {initializeCalled = true;}
    void OnStart() {startCalled = true;}
    void OnStop() {stopCalled = true;}
    void OnStep() {stepCalled = true;}

    bool initializeCalled{false};
    bool startCalled{false};
    bool stopCalled{false};
    bool stepCalled{false};
};

class PrecisionDbgTaskUselessStub : public PrecisionDbgTaskStub {
public:
    bool Condition(const DebuggerConfig& cfg) const override {return false;}
};

TEST(PrecisionDebuggerTest, TestRegisterBeforeInit) {
    PrecisionDebugger& debugger = PrecisionDebugger::GetInstance();
    PrecisionDbgTaskStub stubTask;

    DebuggerConfig::GetInstance().Reset();
    debugger.RegisterDebuggerTask(&stubTask);
    stubTask.Register();

    EXPECT_FALSE(debugger.IsEnable());
    EXPECT_EQ(debugger.GetCurStep(), 0);
    debugger.Start();
    EXPECT_FALSE(debugger.IsEnable());
    debugger.Stop();
    debugger.Step();
    EXPECT_EQ(debugger.GetCurStep(), 0);

    EXPECT_FALSE(stubTask.initializeCalled);
    EXPECT_FALSE(stubTask.startCalled);
    EXPECT_FALSE(stubTask.stopCalled);
    EXPECT_FALSE(stubTask.stepCalled);

    debugger.UnRegisterDebuggerTask(&stubTask);
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
    PrecisionDbgTaskStub stubTask1;
    PrecisionDbgTaskStub stubTask2;
    PrecisionDbgTaskUselessStub stubTask3;
    MOCKER(MindStudioDebugger::AscendCLApi::LoadAclApi)
    .stubs()
    .then(returnValue(0));
    MOCKER(MindStudioDebugger::AscendCLApi::AclApiAclrtSynchronizeDevice)
    .stubs()
    .then(returnValue(0))
    .expects(atLeast(1));

    stubTask1.Register();
    EXPECT_EQ(debugger.Initialize("MindSpore", CONFIG_EXAMPLE), 0);
    stubTask2.Register();
    stubTask3.Register();

    EXPECT_TRUE(stubTask1.initializeCalled);
    EXPECT_TRUE(stubTask2.initializeCalled);
    EXPECT_FALSE(stubTask3.initializeCalled);
    EXPECT_FALSE(stubTask1.startCalled);
    EXPECT_FALSE(stubTask2.stopCalled);
    EXPECT_FALSE(stubTask3.stepCalled);

    debugger.Start();
    EXPECT_TRUE(stubTask1.startCalled);
    EXPECT_FALSE(stubTask3.startCalled);

    debugger.Stop();
    EXPECT_TRUE(stubTask1.stopCalled);
    EXPECT_TRUE(stubTask2.stopCalled);

    debugger.Step();
    EXPECT_TRUE(stubTask1.stepCalled);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

}
