
TOP_DIR							?= /
COMPILER						:= g++
COMPILER_FLAG					:= -fPIC -shared
INC_FLAG						:= -I$(TOP_DIR)/third_party/inc

all: build

build: libascendcl.so libruntime.so libruntime_camodel.so

libascendcl.so: acl.cpp
	$(COMPILER) $(COMPILER_FLAG) $(INC_FLAG) -o $@ $^
	mv $@ ${TOP_DIR}/third_party/lib

libruntime.so: runtime.cpp
	$(COMPILER) $(COMPILER_FLAG) $(INC_FLAG) -o $@ $^
	mv $@ ${TOP_DIR}/third_party/lib

libruntime_camodel.so: runtime.cpp
	$(COMPILER) $(COMPILER_FLAG) $(INC_FLAG) -o $@ $^
	mv $@ ${TOP_DIR}/third_party/lib