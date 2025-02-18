

# Build and Install npu-dynolog-plugin
```
# install pybind11
pip install pybind11

# build dynolog_npu_plugin wheel
python3 setup.py bdist_wheel
# install
pip install dist/{dynolog-npu-plugin-xxx.wheel}

# example
import IPCMonitor
dyno_worker = IPCMonitor.PyDynamicMonitorProxy()
dyno_worker.init_dyno(0)
```
