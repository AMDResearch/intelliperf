# Simple Makefile for scan benchmark
CXX = hipcc
TARGET = scan_bench
SOURCE = scan.hip
CXXFLAGS = -O3 -g

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
