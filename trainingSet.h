//
// Created by damir on 14.09.2019.
//

#ifndef NEURON_TRAININGSET_H
#define NEURON_TRAININGSET_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class trainingSet
{
public:
    explicit trainingSet(std::string filename);
    bool isEOF() { return trainingDataFile.eof(); }
    void getTopology(std::vector<unsigned> &topology);
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);
private:
    std::ifstream trainingDataFile;
};

#endif //NEURON_TRAININGSET_H
