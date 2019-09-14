//
// Created by damir on 14.09.2019.
//

#ifndef NEURON_NET_H
#define NEURON_NET_H

#include <vector>
#include <cstdlib>

class Neuron;

typedef std::vector<Neuron> Layer;

class net {

public:
    explicit net(const std::vector<unsigned> &topology);
    void feedForward (const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError() const { return recentAverageError; }
private:
    std::vector<Layer> layers;
    double error;
    double recentAverageError;
    static double recentAverageSmoothingFactor;
};


#endif //NEURON_NET_H
