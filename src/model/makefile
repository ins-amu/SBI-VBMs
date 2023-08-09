
CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow  

# location of the Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)


all: _mpr_damaged.so 

mpr_damaged: _mpr_damaged.so

OBJS = mpr_damaged_wrap.o 
# -----------------------------------------------------------------------------
_mpr_damaged.so: mpr_damaged_wrap.o
	$(CXX) -shared -fopenmp mpr_damaged_wrap.o -o _mpr_damaged.so

mpr_damaged_wrap.o : mpr_damaged_wrap.cpp mpr_damaged.hpp utility.hpp bold.hpp
	$(CXX) $(CXXFLAGS) $< -I $(PYTHON_INCLUDE)

mpr_damaged_wrap.cpp : mpr_damaged.i 
	swig $(SWIGFLAGS) -o mpr_damaged_wrap.cpp mpr_damaged.i


.PHONEY:clean
clean :
	rm -rf *.o  *.so 
	rm -rf mpr_damaged_wrap.cpp mpr_damaged.py
	rm -rf __pycache__

.PHONEY : clean all mpr_damaged
