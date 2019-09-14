cmake_minimum_required(VERSION 3.15)
project(neuron)

set(CMAKE_CXX_STANDARD 14)

add_executable(neuron main.cpp net.cpp net.h Neuron.cpp Neuron.h trainingSet.cpp trainingSet.h)