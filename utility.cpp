#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

template<typename T>
void writeToTxt(ofstream &file, T arg) {
    file << fixed << setprecision(2) << arg << ' ';
}

template<typename T, typename ...Args>
void writeToTxt(ofstream &file, T arg, Args... args) {
    writeToTxt(file, arg);
    writeToTxt(file, args...);
}
