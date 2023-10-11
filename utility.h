#ifndef CUDASIFT_UTILITY_H
#define CUDASIFT_UTILITY_H

#include <fstream>

using namespace std;

template<typename T>
void writeToTxt(ofstream &file, T arg);

template<typename T, typename ...Args>
void writeToTxt(ofstream &file, T arg, Args... args);

#include "utility.cpp"

#endif //CUDASIFT_UTILITY_H
