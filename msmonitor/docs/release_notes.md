# 版本说明

| msmonitor版本 | 发布日期       | 下载链接                                                                 | 校验码                                                                 | 配套CANN版本 | 配套torch_npu版本 | 配套MindSpore版本 | 
|---------------|----------------|--------------------------------------------------------------------------|------------------------------------------------------------------------|--------------|--------------|--------------|
| 8.1.0         | 2025-07-11     | [aarch64_8.1.0.zip](https://ptdbg.obs.cn-north-4.myhuaweicloud.com/profiler/msmonitor/8.1.0/aarch64_8.1.0.zip) | ce136120c0288291cc0a7803b1efc8c8416c6105e9d54c17ccf2e2510869fada       | 8.1.RC1及以上      | v7.1.0及以上  |  2.7.0-rc1及以上 | 
|               | 2025-07-11     | [x86_8.1.0.zip](https://ptdbg.obs.cn-north-4.myhuaweicloud.com/profiler/msmonitor/8.1.0/x86_8.1.0.zip)           | 097d11c7994793b6389b19259269ceb3b6b7ac5ed77da3949b3f09da2103b7f2        | 8.1.RC1及以上      | v7.1.0及以上  | 2.7.0-rc1及以上 | 

Step 1: 根据aarch64还是x86选择对应安装包链接下载。

Step 2: 校验包完整性

   1. 根据以上下载链接下载包到Linux安装环境。

   2. 进入zip包所在目录，执行如下命令。

      ```
      sha256sum {name}.zip
      ```

      {name}为zip包名称。

      若回显呈现对应版本zip包一致的**校验码**，则表示下载了正确的性能工具zip安装包。示例如下：

      ```bash
      sha256sum aarch64_8.1.0.zip
      ```

Step 3: 包安装（以x86版本为例）

   1. 解压压缩包
   ```bash
   mkdir x86
   unzip x86_8.1.0.zip -d x86
   ```
   
   2. 进入目录
   ```bash 
   cd x86
   ```
   
   3. 安装whl包
   ```bash
   pip install msmonitor_plugin-{mindstudio_version}-cp{python_version}-cp{python_version}-linux_{system_architecture}.whl
   ```
   
   4. 安装dynolog

      有以下三种安装方式可供选择，根据用户服务器系统自行选择： 
      
      方式一：使用deb软件包安装（适用于Debian/Ubuntu等系统）；
      ```
      dpkg -i --force-overwrite dynolog*.deb
      ``` 

      方式二：使用rpm软件包安装（适用于RedHat/Fedora/openSUSE等系统）；
      ```
      rpm -ivh dynolog-*.rpm --nodeps
      ```

      方式三：直接复制bin文件夹到系统中。
