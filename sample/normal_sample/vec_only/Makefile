# Location of the CANN
ASCEND_HOME_PATH			?= /usr/local/Ascend/ascend-toolkit/latest
TOP_DIR						?= /

COMPILER					:= $(ASCEND_HOME_PATH)/compiler/ccec_compiler/bin/ccec
COMPILER_FLAG				:= -xcce -O2 -std=c++17 --cce-aicore-arch=dav-c220-vec
LINK_FLAG					:= --cce-fatobj-link --cce-aicore-arch=dav-c220-vec
ASCENDC_INC_FLAG 			:= -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface -I${ASCEND_HOME_PATH}/include
HOST_INC_FLAG				:= -I${TOP_DIR}/third_party/inc
LINK_LIBS					:= -L${TOP_DIR}/third_party/lib -lruntime -lascendcl -lstdc++

all: build

build: add_kernel.o main.o add.fatbin

add_kernel.o: add_kernel.cpp
	$(COMPILER) $(COMPILER_FLAG) $(ASCENDC_INC_FLAG) -o $@ -c $^

main.o: main.cpp
	$(COMPILER) $(COMPILER_FLAG) $(HOST_INC_FLAG) -o $@ -c $^

add.fatbin: add_kernel.o main.o
	$(COMPILER) $(LINK_FLAG) -o $@ $^ $(LINK_LIBS)

.PHONY: clean
clean:
	rm *.o add.fatbin