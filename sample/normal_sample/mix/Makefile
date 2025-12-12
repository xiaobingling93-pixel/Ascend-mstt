# Location of the CANN, 主要基于${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake中内容简化
ASCEND_HOME_PATH			?= /usr/local/Ascend/ascend-toolkit/latest

COMPILER					:= bisheng # 参考device_config.cmake中CMAKE_C_COMPILER配置
COMPILER_FLAG				:= -xcce -O2 -std=c++17
LINK_FLAG					:= --cce-fatobj-link
DAV_FLAG                    := --cce-aicore-arch=dav-c220
ASCENDC_INC_FLAG 			:= -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface -I${ASCEND_HOME_PATH}/include # 参考device_intf.cmake的配置简化
TILING_INC_FLAG				:= -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
HOST_INC_FLAG				:= -I${ASCEND_HOME_PATH}/include
LINK_LIBS					:= -L${ASCEND_HOME_PATH}/lib64 -lruntime -lascendcl -lstdc++ -lprofapi -lmmpa -lascendalog -lregister -lerror_manager -lc_sec
LINK_STATIC_LIBS			:= ${ASCEND_HOME_PATH}/lib64/libascendc_runtime.a

all: build

build: matmul_leakyrelu_kernel.o main.o matmul_leakyrelu.fatbin

matmul_leakyrelu_kernel.o: matmul_leakyrelu_kernel.cpp
	$(COMPILER) $(COMPILER_FLAG) $(DAV_FLAG) $(ASCENDC_INC_FLAG) -o $@ -c $^

main.o: main.cpp
	$(COMPILER) $(COMPILER_FLAG) $(DAV_FLAG) $(HOST_INC_FLAG) $(TILING_INC_FLAG) -o $@ -c $^

matmul_leakyrelu.fatbin: matmul_leakyrelu_kernel.o main.o
	$(COMPILER) $(LINK_FLAG) $(DAV_FLAG) -o $@ $^ $(LINK_LIBS) $(LINK_STATIC_LIBS)

.PHONY: clean
clean:
	rm *.o *.fatbin