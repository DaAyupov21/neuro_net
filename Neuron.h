//
// Created by damir on 14.09.2019.
//

#ifndef NEURON_NEURON_H
#define NEURON_NEURON_H

#include <vector>
#include <cstdlib>
#include "net.h"

struct Connection
{
    double weight;
    double deltaWeight;
};
class Neuron {
public:
    Neuron (unsigned numOutputs, unsigned myIndex);
    void setOutputVal (double val) {outputVals = val; }
    double getOutputVal() const { return outputVals; }
    void feedForward(const Layer &prewLayer);
    void calcOutputGradiens(double targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);


private:
    double outputVals;
    std::vector<Connection> outputWeights;
    unsigned m_myIndex;
    double gradient;
    static double eta;
    static double alpha;
    double sumDOW(const Layer &nextLayer) const;

    static double randomWeight() {return rand() / double(RAND_MAX); }
    static double actvationFunction(double x);
    static double actvationFunctionDerivative(double x);





};


#endif //NEURON_NEURON_H
