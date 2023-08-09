
%module mpr_damaged

%{
#include "mpr_damaged.hpp"
%}

%include stl.i
%include <std_vector.i>
%include "std_string.i"
/* instantiate the required template specializations */
namespace std {
    %template(IntVector)     std::vector<int>;
    %template(DoubleVector)  std::vector<double>;
    %template(DoubleVector2) std::vector<vector<double> >;
    %template(SingleVector)  std::vector<float>;
    %template(SingleVector2) std::vector<vector<float> >;
}

%include "mpr_damaged.hpp"
