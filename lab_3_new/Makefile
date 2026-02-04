ifeq ($(CXXFLAGS), )
CXXFLAGS = -O3
endif
ifeq ($(CXX), )
CXX=g++
endif
ENV=CXX=$(CXX) OUTDIR=../bin/ "CXXFLAGS=$(CXXFLAGS)"

.phony: all run clean

all:bin/run ;

bin/run:
	make -C parallel_mod/ ../bin/run $(ENV)
run: bin/run ;

clean:
	make -C parallel_mod/ clean $(ENV)
