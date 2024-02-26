# Location of the CANN
ASCEND_HOME_PATH			?= /usr/local/Ascend/ascend-toolkit/latest
TOP_DIR						?= /

COMPILER					:= $(ASCEND_HOME_PATH)/compiler/ccec_compiler/bin/ccec
COMPILER_FLAG				:= -xcce -O2 -std=c++17
LINK_FLAG					:= --cce-fatobj-link
DAV_FLAG                    := --cce-aicore-arch=dav-c220-cube
ASCENDC_INC_FLAG 			:= -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface -I${ASCEND_HOME_PATH}/include
TILING_INC_FLAG				:= -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
HOST_INC_FLAG				:= -I${TOP_DIR}/third_party/inc
LINK_LIBS					:= -L${TOP_DIR}/third_party/lib -lruntime -lascendcl -lstdc++

all: build

build: matmul_kernel.o main.o matmul.fatbin

matmul_kernel.o: matmul_kernel.cpp
	$(COMPILER) $(COMPILER_FLAG) $(DAV_FLAG) $(ASCENDC_INC_FLAG) -o $@ -c $^

main.o: main.cpp
	$(COMPILER) $(COMPILER_FLAG) $(DAV_FLAG) $(HOST_INC_FLAG) $(TILING_INC_FLAG) -o $@ -c $^

matmul.fatbin: matmul_kernel.o main.o
	$(COMPILER) $(LINK_FLAG) $(DAV_FLAG) -o $@ $^ $(LINK_LIBS)

.PHONY: clean
clean:
    rm *.o *.fatbin