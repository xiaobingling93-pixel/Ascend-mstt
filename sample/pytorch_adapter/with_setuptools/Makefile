# Location of the CANN, 主要基于${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake中内容简化
ASCEND_HOME_PATH			?= /usr/local/Ascend/ascend-toolkit/latest

COMPILER					:= $(ASCEND_HOME_PATH)/compiler/ccec_compiler/bin/ccec  # 参考device_config.cmake中CMAKE_C_COMPILER配置
COMPILER_FLAG				:= -xcce -O2 -std=c++17
DYNAMIC_LIB_FLAG			:= -fPIC -shared
DAV_FLAG                    := --cce-aicore-arch=dav-c220-vec
ASCENDC_INC_FLAG 			:= -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface -I${ASCEND_HOME_PATH}/include  # 参考device_intf.cmake的配置简化

all: build

build: libcustom_kernels.so

# 后续如果要扩展，把多个kernel的cpp都加到后面
libcustom_kernels.so: add_kernel.cpp
	$(COMPILER) $(DYNAMIC_LIB_FLAG) $(COMPILER_FLAG) $(DAV_FLAG) $(ASCENDC_INC_FLAG) -o $@ $^

.PHONY: clean
clean:
	rm *.so