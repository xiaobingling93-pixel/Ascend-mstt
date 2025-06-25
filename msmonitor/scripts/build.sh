#!/bin/bash
set -e
export BUILD_PROMETHEUS=0
export BUILD_TENSORBOARD=1
export USE_TENSORBOARD="OFF"

# 设置 CARGO_HOME
export CARGO_HOME="/root/.cargo"

# 创建 Cargo 配置目录
mkdir -p ${CARGO_HOME}

# 创建 config.toml(安全编译选项)
cat > ${CARGO_HOME}/config.toml << EOF
[net]
git-fetch-with-cli = true

[build]
rustflags = [
    "-C", "relocation_model=pie",
    "-C", "link-args=-Wl,-z,now",
    "-C", "link-args=-Wl,-z,relro",
    "-C", "strip=symbols",
    "-C", "overflow_checks",
    "-C", "link-args=-static-libgcc",
    "-C", "link-args=-static-libstdc++"
]
EOF

check_gcc_version() {
    if ! command -v gcc >/dev/null 2>&1; then
        echo "ERROR: gcc command not found"
        return 1
    fi

    local GCC_VERSION=$(gcc -dumpversion)
    local GCC_MAJOR=$(echo $GCC_VERSION | cut -d. -f1)
    local GCC_MINOR=$(echo $GCC_VERSION | cut -d. -f2)

    if [ "$GCC_MAJOR" -lt 8 ] || ([ "$GCC_MAJOR" -eq 8 ] && [ "$GCC_MINOR" -lt 5 ]); then
        echo "ERROR: gcc version must be greater than or equal to 8.5.0"
        echo "Current gcc version: $GCC_VERSION"
        return 1
    fi
    echo "Check pass: current gcc version is $GCC_VERSION"
    return 0
}

check_rust_version() {
    if ! command -v rustc >/dev/null 2>&1; then
        echo "ERROR: rustc command not found"
        return 1
    fi

    local RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    local RUST_MAJOR=$(echo $RUST_VERSION | cut -d. -f1)
    local RUST_MINOR=$(echo $RUST_VERSION | cut -d. -f2)

    if [ "$RUST_MAJOR" -lt 1 ] || ([ "$RUST_MAJOR" -eq 1 ] && [ "$RUST_MINOR" -lt 81 ]); then
        echo "ERROR: Rust version must be greater than or equal to 1.81"
        echo "Current Rust version: $RUST_VERSION"
        return 1
    fi
    echo "Check pass: current Rust version is $RUST_VERSION"
    return 0
}

update_and_checkout_submodule() {
    DYNLOG_COMMIT_ID="a9b6aeddcd6363252f5388cb0dd942981a09a24b"

    git submodule update --init
    if [ $? -ne 0 ]; then
        echo "ERROR: update git submodule failed"
        return 1
    fi

    cd ./third_party/dynolog
    git checkout ${DYNLOG_COMMIT_ID}

    if [ "${BUILD_TENSORBOARD}" -ne 0 ]; then
        if [ ! -d "./third_party/tensorboard_logger" ]; then
            git submodule add https://github.com/RustingSword/tensorboard_logger.git ./third_party/tensorboard_logger
        fi
        USE_TENSORBOARD="ON"
    fi
    git submodule update --init --recursive

    if [ $? -ne 0 ]; then
        echo "ERROR: switch to dynolog specified commit failed"
        cd ..
        return 1
    fi
    echo "Check pass: switch to dynolog specified commit ${DYNLOG_COMMIT_ID}"
    cd ../../
    return 0
}

PACKAGE_TYPE=""
while getopts "t:" opt; do
    case $opt in
        t)
            PACKAGE_TYPE="$OPTARG"
            if [[ "$PACKAGE_TYPE" != "deb" && "$PACKAGE_TYPE" != "rpm" ]]; then
                echo "ERROR: Invalid package type. Supported types: deb, rpm"
                exit 1
            fi
            ;;
        \?)
            echo "Usage: $0 [-t package_type]"
            echo "package_type: deb or rpm (optional, if not specified will only build)"
            exit 1
            ;;
    esac
done

echo "------------------ Check GCC and Rust version ----------------------"
check_gcc_version
check_rust_version

echo "------------------ Update and checkout submodule -------------------"
update_and_checkout_submodule

cp -r dynolog_npu/cmake third_party/dynolog

echo "------------------ Generate patch for Ascend -----------------------"
bash scripts/gen_dyno_patches.sh

echo "------------------ Apply patch for Ascend --------------------------"
bash scripts/apply_dyno_patches.sh

echo "------------------ Build dynolog patch for Ascend-------------------"
cd third_party/dynolog
rm -rf build
if [ -z "$PACKAGE_TYPE" ]; then
    bash scripts/build.sh
    echo "Build dynolog success without packaging"
elif [ "$PACKAGE_TYPE" = "deb" ]; then
    ARCHITECTURE=$(uname -m)
    CONTROL_FILE="scripts/debian/control"
    ARCH="amd64"
    if [[ "$ARCHITECTURE" == "aarch64" ]]; then
        sed -i 's/^Architecture: .*/Architecture: arm64/' "$CONTROL_FILE"
        ARCH="arm64"
        echo "dpkg Architecture set to arm64"
    fi
    export ARCH=$ARCH
    bash scripts/debian/make_deb.sh
    unset ARCH
    mv dynolog_*.deb ../../
    echo "Build dynolog deb package success"
elif [ "$PACKAGE_TYPE" = "rpm" ]; then
    bash scripts/rpm/make_rpm.sh
    mv dynolog-*.rpm ../../
    echo "Build dynolog rpm package success"
fi
