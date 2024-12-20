#include <cmath>
#include <gtest/gtest.h>

#include "test_utils.hpp"
#include "utils/CPythonUtils.hpp"

using namespace MindStudioDebugger;
using namespace MindStudioDebugger::CPythonUtils;

namespace MsProbeTest {

class CPythonUtilsTest : public ::testing::Test {
protected:
    void SetUp() override {
        Py_Initialize();
    }

    void TearDown() override {
        Py_Finalize();
    }
};

TEST_F(CPythonUtilsTest, CPythonAgent) {
    PythonObject obj = PythonObject::From("test");
    std::string name = "test_object";
    int32_t result = RegisterPythonObject(name, obj);
    EXPECT_EQ(result, 0);
    bool registerd = IsPyObjRegistered(name);
    EXPECT_TRUE(registerd);

    result = RegisterPythonObject(name, obj);
    EXPECT_EQ(result, -1);
    registerd = IsPyObjRegistered(name);
    EXPECT_TRUE(registerd);

    name = "test_object";
    UnRegisterPythonObject(name);
    name = "test_object1";
    UnRegisterPythonObject(name);
    registerd = IsPyObjRegistered(name);
    EXPECT_FALSE(registerd);

    result = RegisterPythonObject(name, obj);
    EXPECT_EQ(result, 0);
    registerd = IsPyObjRegistered(name);
    EXPECT_TRUE(registerd);

    PythonObject registerd_obj = GetRegisteredPyObj(name);
    EXPECT_EQ(static_cast<PyObject*>(registerd_obj), static_cast<PyObject*>(obj));
    EXPECT_TRUE(registerd_obj.IsString());
    EXPECT_EQ(registerd_obj.ToString(), "test");

    PythonObject invalid_obj = GetRegisteredPyObj("invalid_name");
    EXPECT_TRUE(invalid_obj.IsNone());
}

TEST_F(CPythonUtilsTest, PythonObjectFromTo) {
    // 测试PythonObject的From和To函数
    int32_t input_int = -42;
    PythonObject obj_int = PythonObject::From(input_int);
    EXPECT_TRUE(obj_int.IsNumber());

    int32_t output_int;
    EXPECT_EQ(obj_int.To(output_int), 0);
    EXPECT_EQ(output_int, input_int);

    uint32_t input_uint = 56;
    PythonObject obj_uint = PythonObject::From(input_uint);
    EXPECT_TRUE(obj_uint.IsNumber());

    uint32_t output_uint;
    EXPECT_EQ(obj_uint.To(output_uint), 0);
    EXPECT_EQ(output_uint, input_uint);

    double input_double = 3.14;
    PythonObject obj_double = PythonObject::From(input_double);
    EXPECT_TRUE(obj_double.IsNumber());

    double output_double;
    EXPECT_EQ(obj_double.To(output_double), 0);
    EXPECT_DOUBLE_EQ(output_double, input_double);

    std::string input_str = "hello";
    PythonObject obj_str = PythonObject::From(input_str);
    EXPECT_TRUE(obj_str.IsString());

    std::string output_str;
    EXPECT_EQ(obj_str.To(output_str), 0);
    EXPECT_EQ(output_str, input_str);

    const char* input_char = "world";
    PythonObject obj_str1 = PythonObject::From(input_char);
    EXPECT_TRUE(obj_str1.IsString());

    EXPECT_EQ(obj_str1.To(output_str), 0);
    EXPECT_EQ(output_str, std::string(input_char));

    bool input_bool = true;
    PythonObject obj_bool = PythonObject::From(input_bool);
    EXPECT_TRUE(obj_bool.IsBool());

    bool output_bool;
    EXPECT_EQ(obj_bool.To(output_bool), 0);
    EXPECT_EQ(output_bool, input_bool);

    std::vector<int> input_vector_int = {1, 2, 3, 100};
    PythonObject list_int_obj = PythonObject::From(input_vector_int);
    EXPECT_TRUE(list_int_obj.IsList());

    std::vector<int> output_vector_int;
    EXPECT_EQ(list_int_obj.To(output_vector_int), 0);

    size_t size = input_vector_int.size();
    EXPECT_EQ(size, output_vector_int.size());

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(input_vector_int[i], output_vector_int[i]);
    }

    std::vector<std::string> input_vector_str = {"a", "bb", "ccc", "dddd"};
    PythonObject list_str_obj = PythonObject::From(input_vector_str);
    EXPECT_TRUE(list_str_obj.IsList());

    std::vector<std::string> output_vector_str;
    EXPECT_EQ(list_str_obj.To(output_vector_str), 0);

    size = input_vector_str.size();
    EXPECT_EQ(size, output_vector_str.size());

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(input_vector_str[i], output_vector_str[i]);
    }
}

TEST_F(CPythonUtilsTest, PythonObjectImport) {
    PythonObject sys = PythonObject::Import("sys");
    EXPECT_TRUE(sys.IsModule());
    EXPECT_EQ(static_cast<PyObject*>(sys), PyImport_ImportModule("sys"));
    EXPECT_FALSE(sys.IsNone());
    PythonObject invalid = PyImport_ImportModule("invalid");
    EXPECT_TRUE(invalid.IsNone());
}

TEST_F(CPythonUtilsTest, PythonObjectGetAttr) {
    PythonObject sys = PythonObject::Import("sys");
    PythonObject sys_path = sys.Get("path");
    EXPECT_TRUE(sys_path.IsList());
    PythonObject fexit = sys.Get("exit");
    EXPECT_TRUE(fexit.IsCallable());
    PythonObject invalid = sys.Get("invalid");
    EXPECT_TRUE(invalid.IsNone());

    std::vector<int> input_vector = {1, 2, 3, 100};
    PythonObject list_obj = PythonObject::From(input_vector);
    PythonObject append = list_obj.Get("append");
    EXPECT_TRUE(append.IsCallable());
}

TEST_F(CPythonUtilsTest, PythonObjectCall) {
    PythonObject int_class = PythonObject::Import("builtins").Get("int");
    EXPECT_TRUE(int_class.IsCallable());
    PythonObject int_obj = int_class.Call();
    EXPECT_TRUE(int_obj.IsNumber());
    int result = -1;
    EXPECT_EQ(int_obj.To(result), 0);
    EXPECT_EQ(result, 0);

    PythonObject ret = PythonObject::Import("builtins").Call();
    EXPECT_TRUE(ret.IsNone());
}

TEST_F(CPythonUtilsTest, PythonObjectType) {
    PythonObject none = Py_None;
    EXPECT_TRUE(none.IsNone());
    EXPECT_FALSE(none.IsNumber() || none.IsCallable());

    PythonObject pytrue = Py_True;
    EXPECT_TRUE(pytrue.IsBool());
    EXPECT_FALSE(pytrue.IsString() || pytrue.IsCallable());

    PythonObject builtins = PyImport_ImportModule("builtins");
    EXPECT_TRUE(builtins.IsModule());
    EXPECT_FALSE(builtins.IsList() || builtins.IsCallable());

    PythonObject int_class = builtins.Get("int");
    EXPECT_TRUE(int_class.IsCallable());
    EXPECT_FALSE(builtins.IsDict());

    PythonObject dict = builtins.Get("__dict__");
    EXPECT_TRUE(dict.IsDict());
    EXPECT_FALSE(dict.IsNone() || dict.IsCallable());
}

TEST_F(CPythonUtilsTest, PythonNumberObject) {
    PythonNumberObject o1(PyLong_FromLong(123));
    PythonNumberObject o2(PyFloat_FromDouble(3.14));
    PythonNumberObject o3 = PythonNumberObject::From(321);
    PythonNumberObject o4 = PythonNumberObject::From(2.33);
    PythonNumberObject o5(PythonObject::From(4.44));
    PythonNumberObject o6(PythonObject::From("1111"));

    int int_v;
    EXPECT_EQ(o1.To(int_v), 0);
    EXPECT_EQ(int_v, 123);
    double double_v;
    EXPECT_EQ(o2.To(double_v), 0);
    EXPECT_TRUE(std::fabs(double_v - 3.14) < 1e-5);
    EXPECT_EQ(o3.To(int_v), 0);
    EXPECT_EQ(int_v, 321);
    EXPECT_EQ(o4.To(double_v), 0);
    EXPECT_TRUE(std::fabs(double_v - 2.33) < 1e-5);
    EXPECT_EQ(o5.To(double_v), 0);
    EXPECT_TRUE(std::fabs(double_v - 4.44) < 1e-5);
    EXPECT_TRUE(o6.IsNone());
}

TEST_F(CPythonUtilsTest, PythonStringObject) {
    PythonStringObject o1(PyUnicode_FromString("hello"));
    PythonStringObject o2 = PythonStringObject::From("OK");
    PythonStringObject o3 = PythonStringObject::From(std::string("banana"));
    PythonStringObject o4(PythonObject::From(1));

    EXPECT_EQ(o1.ToString(), "hello");
    EXPECT_EQ(o2.ToString(), "OK");
    EXPECT_EQ(o3.ToString(), "banana");
    EXPECT_TRUE(o4.IsNone());
}

TEST_F(CPythonUtilsTest, PythonBoolObject) {
    PythonBoolObject o1(Py_True);
    PythonBoolObject o2(Py_False);
    PythonBoolObject o3(PythonObject::From(true));
    PythonBoolObject o4(PythonObject::From(0));

    EXPECT_EQ(o1, true);
    EXPECT_EQ(o2, false);
    EXPECT_EQ(o3, true);
    EXPECT_TRUE(o4.IsNone());
}

TEST_F(CPythonUtilsTest, PythonListObject) {
    PythonListObject empty_list(5);
    PythonListObject sys_path(static_cast<PyObject*>(PythonObject::Import("sys").Get("path")));
    PythonListObject list1 = PythonListObject::From(std::vector<int>({1, 3, 5, 7}));
    PythonListObject list2 = PythonListObject::From(std::vector<std::vector<int>>({{1, 3, 5, 7}, {2, 4, 6}}));
    PythonListObject list3;

    int val;
    EXPECT_EQ(empty_list.Size(), 5);
    EXPECT_FALSE(sys_path.IsNone());
    EXPECT_TRUE(sys_path.Size() > 0);
    EXPECT_TRUE(sys_path.GetItem(0).IsString());
    EXPECT_EQ(list1.Size(), 4);
    EXPECT_EQ(list1.GetItem(1).To(val), 0);
    EXPECT_EQ(val, 3);
    EXPECT_EQ(list1.GetItem(3).ToString(), "7");
    EXPECT_TRUE(list1.GetItem(4).IsNone());
    EXPECT_EQ(list2.Size(), 2);
    EXPECT_TRUE(list2.GetItem(0).IsList());
    EXPECT_EQ(list2.GetItem(1).ToString(), "[2, 4, 6]");
    EXPECT_EQ(list3.Size(), 0);
    list3.Append(PythonObject::From(1));
    EXPECT_EQ(list3.Size(), 1);
    list3.Append(PythonObject::From("2")).Append(PythonObject::From(true));
    EXPECT_EQ(list3.Size(), 3);
    EXPECT_EQ(list3.GetItem(1).ToString(), "2");
    list3.SetItem(1, empty_list);
    EXPECT_EQ(list3.Size(), 3);
    EXPECT_EQ(static_cast<PyObject*>(list3.GetItem(1)), static_cast<PyObject*>(empty_list));
    list3.Insert(0, sys_path);
    EXPECT_EQ(list3.Size(), 4);
    EXPECT_EQ(static_cast<PyObject*>(list3.GetItem(0)), static_cast<PyObject*>(sys_path));
    PythonTupleObject tuple = list3.ToTuple();
    EXPECT_FALSE(tuple.IsNone());
}

TEST_F(CPythonUtilsTest, PythonTupleObject) {
    PythonTupleObject tuple1;
    PythonTupleObject tuple2(PyTuple_New(0));
    PythonTupleObject tuple3 = PythonTupleObject::From(std::vector<std::string>({"ab", "cd"}));
    PythonTupleObject tuple4 = PythonListObject::From(std::vector<int>({1, 3, 5})).ToTuple();

    EXPECT_FALSE(tuple1.IsNone());
    EXPECT_EQ(tuple1.Size(), 0);
    EXPECT_TRUE(tuple1.GetItem(0).IsNone());
    EXPECT_FALSE(tuple2.IsNone());
    EXPECT_EQ(tuple2.Size(), 0);
    EXPECT_EQ(tuple3.Size(), 2);
    EXPECT_EQ(tuple3.GetItem(0).ToString(), "ab");
    EXPECT_EQ(tuple4.Size(), 3);
    EXPECT_EQ(tuple4.GetItem(0).ToString(), "1");
}

TEST_F(CPythonUtilsTest, PythonDictObject) {
    PythonDictObject dict1;
    PythonDictObject dict2(PyDict_New());
    PythonDictObject dict3 = PythonDictObject::From(std::map<int, std::string>({{1, "a"}, {2, "b"}}));

    EXPECT_FALSE(dict1.IsNone());
    EXPECT_FALSE(dict2.IsNone());
    EXPECT_TRUE(dict2.GetItem("none").IsNone());
    EXPECT_FALSE(dict3.IsNone());
    EXPECT_EQ(dict3.GetItem(1).ToString(), "a");
    EXPECT_EQ(dict3.GetItem(2).ToString(), "b");
    EXPECT_TRUE(dict3.GetItem(3).IsNone());
    dict3.Add(std::string("apple"), std::string("banana"));
    EXPECT_EQ(dict3.GetItem(std::string("apple")).ToString(), "banana");
    dict3.Delete(std::string("apple"));
    EXPECT_TRUE(dict3.GetItem(std::string("apple")).IsNone());
}

}
