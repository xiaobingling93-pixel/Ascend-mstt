# msMonitor安装
## 下载软件包安装（推荐）
最新的预编译安装包和版本依赖请查看[msMonitor release](./release_notes.md)，并根据指导进行校验和安装。

## 源码编译安装

### 1. clone 代码

```bash
git clone https://gitcode.com/Ascend/mstt.git
```

### 2. 安装依赖
dynolog的编译依赖，确保安装了以下依赖：
<table>
  <tr>
   <td>Language
   </td>
   <td>Toolchain
   </td>
  </tr>
  <tr>
   <td>C++
   </td>
   <td>gcc 8.5.0+
   </td>
  </tr>
  <tr>
   <td>Rust
   </td>
   <td>Rust >= 1.81
   </td>
  </tr>
</table>

- 安装rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source $HOME/.cargo/env
```

- 安装ninja

```bash
# debian
sudo apt-get install -y cmake ninja-build

# centos
sudo yum install -y cmake ninja
```

- 安装protobuf (tensorboard_logger三方依赖，用于对接tensorboard展示)
- **说明**：要求protobuf版本为3.12或更高版本
```bash
# debian
sudo apt install -y protobuf-compiler libprotobuf-dev

# centos
sudo yum install -y protobuf protobuf-devel protobuf-compiler

# Python
pip install protobuf
```

- （可选）安装openssl（RPC TLS认证）& 生成证书密钥
- **说明**：如果不需要使用TLS证书密钥加密，该步骤可跳过。

```bash
# debian
sudo apt-get install -y openssl

# centos
sudo yum install -y openssl
```
dyno CLI与dynolog daemon之间的RPC通信使用TLS证书密钥加密，在启动dyno和dynolog二进制时可以指定证书密钥存放的路径，路径下需要满足如下结构和名称。
**用户应使用与自己需求相符的密钥生成和存储机制，并保证密钥安全性与机密性。**

服务端证书目录结构： 
```bash
server_certs
├── ca.crt (根证书，用于验证其他证书的合法性，必选)
├── server.crt (服务器端的证书，用于向客户端证明服务器身份，必选)
├── server.key (服务器端的私钥文件，与server.crt配对使用，支持加密，必选)
└── ca.crl (证书吊销列表，包含已被吊销的证书信息，可选)
```
客户端证书目录结构：
```bash
client_certs
├── ca.crt (根证书，用于验证其他证书的合法性，必选)
├── client.crt (客户端证书，用于向服务器证明客户端身份，必选)
├── client.key (客户端的私钥文件，与client.crt配对使用，支持加密，必选)
└── ca.crl (证书吊销列表，包含已被吊销的证书信息，可选)
```

### 3. 编译

- dynolog编译

默认编译生成dyno和dynolog二进制文件，-t参数可以支持将二进制文件打包成deb包或rpm包。

```bash
# 编译dyno和dynolog二进制文件
bash scripts/build.sh

# 编译deb包, 当前支持amd64和aarch64平台, 默认为amd64, 编译aarch64平台需要修改third_party/dynolog/scripts/debian/control文件中的Architecture改为arm64
bash scripts/build.sh -t deb

# 编译rpm包, 当前只支持amd64平台
bash scripts/build.sh -t rpm
```

- msmonitor-plugin wheel包编译

msmonitor-plugin wheel包提供IPCMonitor，MsptiMonitor等公共能力，使用nputrace和npumonitor功能前必须安装该wheel包，具体编译安装指导可参考[msmonitor-plugin编包指导](../plugin/README.md)。