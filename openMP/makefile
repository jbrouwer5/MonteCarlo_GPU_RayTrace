
CXX=g++

CXXFLAGS=-std=c++20 -O3 -march=native -mtune=native -fopenmp

TARGET=raytrace

SOURCES=parallel.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@