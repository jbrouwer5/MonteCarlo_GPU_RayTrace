
NVCC = nvcc

NVCCFLAGS = -arch=sm_70 -O3 -use_fast_math

TARGET = raytrace

SOURCES = gpu.cu

OBJECTS = $(SOURCES:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(OBJECTS) $(TARGET)