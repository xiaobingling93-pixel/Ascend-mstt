

# Plugins for msMonitor
## 模块说明
### IPCMonitor
提供IPC(Inter-Process Communication)通信接口，用于实现
1. IPC控制通道: profiler backend向dynolog daemon获取profiler配置


__PyDynamicMonitorProxy__:  
* `init_dyno` 向dynolog daemon发送注册请求
  * input: npuId(int)
  * return： None
* `poll_dyno` 向dynolog daemon获取Profiler控制参数
  * input: None
  * return: str, 返回控制参数

## 安装方式
### 1. 通过shell脚本一键安装
```
chmod +x build.sh
./build.sh
```
### 2. 手动安装
* 安装依赖
```
pip install wheel
pip install pybind11
```
* 编译whl包
```
python3 setup.py bdist_wheel
```
以上命令执行完成后在plugn/dist目录下生成msMonitor插件whl安装包msmonitor-plugin-{version}.whl
* 安装
```
pip install dist/{msmonitor-plugin-{version}.whl}
```
* 卸载
```
pip uninstall msmonitor-plugin
```