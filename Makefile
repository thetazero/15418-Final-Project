CFLAGS := -std=c++14 -fvisibility=hidden -lpthread -O2 -fopenmp
ISPC=ispc
ISPCTARGET=avx1-i32x8
# ISPCTARGET=sse4
ISPCFLAGS=-O2 --target=$(ISPCTARGET) --arch=x86-64

OBJDIR=src/engine/objs
ENGINEDIR := src/engine
SOURCES := $(ENGINEDIR)/*.cpp
HEADERS := $(ENGINEDIR)/*.h
OBJS=$(OBJDIR)/*.o

TARGETS := eval_ispc run_engine engine_vs_random engine_vs_engine eval_board profile
all: $(TARGETS)

eval_ispc: $(ENGINEDIR)/eval.ispc
	rm -rf $(OBJDIR); mkdir $(OBJDIR)
	$(ISPC) $(ISPCFLAGS) $(ENGINEDIR)/eval.ispc -o $(OBJDIR)/eval_ispc.o -h $(OBJDIR)/eval_ispc.h

run_engine: $(HEADERS) $(SOURCES) src/run_engine.cpp
	$(CXX) -o $@ $(CFLAGS) src/run_engine.cpp $(SOURCES) $(OBJS)

engine_vs_random: $(HEADERS) $(SOURCES) src/engine_vs_random.cpp
	$(CXX) -o $@ $(CFLAGS) src/engine_vs_random.cpp $(SOURCES) $(OBJS)

engine_vs_engine: $(HEADERS) $(SOURCES) src/engine_vs_engine.cpp
	$(CXX) -o $@ $(CFLAGS) src/engine_vs_engine.cpp $(SOURCES) $(OBJS)

eval_board: $(HEADERS) $(SOURCES) src/eval_board.cpp
	$(CXX) -o $@ $(CFLAGS) src/eval_board.cpp $(SOURCES) $(OBJS)

profile: $(HEADERS) $(SOURCES) src/profile.cpp
	$(CXX) -o ./prof/$@ $(CFLAGS) src/profile.cpp $(SOURCES) $(OBJS)

clean: 
	rm -f $(TARGETS)

format:
	clang-format -i src/engine/*.cpp src/engine/*.h src/*.cpp src/engine/*.ispc