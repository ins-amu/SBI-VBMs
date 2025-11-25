
CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow  

# Automatically detect Python configuration
PYTHON_CONFIG := $(shell which python3-config || which python-config)
PYTHON_INCLUDE := $(shell $(PYTHON_CONFIG) --includes 2>/dev/null || python3 -c "from sysconfig import get_paths; print('-I' + get_paths()['include'])")
PYTHON_LDFLAGS := $(shell $(PYTHON_CONFIG) --ldflags 2>/dev/null || echo "")


all: _mpr_damaged.so 

mpr_damaged: _mpr_damaged.so

OBJS = mpr_damaged_wrap.o 
# -----------------------------------------------------------------------------
_mpr_damaged.so: mpr_damaged_wrap.o
	$(CXX) -shared -fopenmp mpr_damaged_wrap.o -o _mpr_damaged.so

mpr_damaged_wrap.o : mpr_damaged_wrap.cpp mpr_damaged.hpp utility.hpp bold.hpp
	$(CXX) $(CXXFLAGS) $< $(PYTHON_INCLUDE)

mpr_damaged_wrap.cpp : mpr_damaged.i 
	swig $(SWIGFLAGS) -o mpr_damaged_wrap.cpp mpr_damaged.i


.PHONEY:clean
clean :
	rm -rf *.o  *.so 
	rm -rf mpr_damaged_wrap.cpp mpr_damaged.py
	rm -rf __pycache__

.PHONEY : clean all mpr_damaged
