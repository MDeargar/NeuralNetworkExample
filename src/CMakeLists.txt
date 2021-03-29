cmake_minimum_required(VERSION 2.8)

project("Compile sources")
add_library(KNNActivationFunc activationfunc2.cpp)
add_library(KNNLayer layer2.cpp)
add_library(KNNNeuralNetwork neuralnetwork2.cpp)
add_library(KNNGeneral general.cpp)
add_library(KNNDataset dataset.cpp)
target_link_libraries(KNNNeuralNetwork KNNActivationFunc KNNLayer KNNGeneral KNNDataset)
