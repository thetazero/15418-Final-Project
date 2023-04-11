CFLAGS := -std=c++14 -fvisibility=hidden -lpthread

SOURCES := src/engine/*.cpp
HEADERS := src/engine/*.h

TARGETS := run_engine engine_vs_random engine_vs_engine
all: $(TARGETS)

run_engine: $(HEADERS) $(SOURCES) src/run_engine.cpp
	$(CXX) -o $@ $(CFLAGS) src/run_engine.cpp $(SOURCES)

engine_vs_random: $(HEADERS) $(SOURCES) src/engine_vs_random.cpp
	$(CXX) -o $@ $(CFLAGS) src/engine_vs_random.cpp $(SOURCES)

engine_vs_engine: $(HEADERS) $(SOURCES) src/engine_vs_engine.cpp
	$(CXX) -o $@ $(CFLAGS) src/engine_vs_engine.cpp $(SOURCES)

clean: 
	rm -f $(TARGETS)

format:
	clang-format -i src/engine/*.cpp src/engine/*.h src/*.cpp