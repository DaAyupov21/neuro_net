//
// Created by damir on 14.09.2019.
//

#include "Neuron.h"
#include "net.h"
#include <cmath>

double Neuron::eta = 0.15; //net learning rate
double Neuron::alpha = 0.5; //momentum/динамика

Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
    for (unsigned c = 0; c < numOutputs; c++){
        outputWeights.push_back(Connection());
        outputWeights.back().weight=randomWeight();
    }
    m_myIndex = myIndex;
}
void Neuron::feedForward(const Layer &prewLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prewLayer.size(); n++){
        sum+=prewLayer[n].getOutputVal()*prewLayer[n].outputWeights[m_myIndex].weight;
    }
    outputVals = Neuron::actvationFunction(sum);
}
double Neuron::actvationFunction(double x) {
    return tanh(x);
}
double Neuron::actvationFunctionDerivative(double x) {
    return 1.0 - x*x;
}
void Neuron::calcOutputGradiens(double targetVals) {
    double delta = targetVals - outputVals;
    gradient = delta*Neuron::actvationFunctionDerivative(outputVals);
}

double Neuron::sumDOW(const Layer &nextLayer) const {
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += outputWeights[n].weight * nextLayer[n].gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer) {
    double dow = sumDOW(nextLayer);
    gradient = dow * Neuron::actvationFunctionDerivative(outputVals);
}

void Neuron::updateInputWeights(Layer &prevLayer) {
    for (unsigned n = 0; n <prevLayer.size(); n++ ){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
        neuron.outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron,outputWeights[m_myIndex].weight += newDeltaWeight;

    }
}
