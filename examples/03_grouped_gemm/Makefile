# Makefile for Grouped GEMM benchmark

CXX = hipcc
CXXFLAGS = -O3 -std=c++17
TARGET = grouped_gemm_bench

all: $(TARGET)

$(TARGET): main.cpp kernel.hip
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

