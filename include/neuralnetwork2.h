#ifndef NEURALNETWORK2_H
#define NEURALNETWORK2_H

#include "dataset.h"
#include "general.h"
#include "layer2.h"
#include <bits/stdc++.h>
#include <tuple>
#include <utility>

namespace KNet {
class NeuralNetwork {
    std::vector<std::pair<std::vector<Layer *>, Layer::Types>> Stratums;
    // layer params (feature size, rows, cols, layer type, activation func, [loss_func]), connection params
    std::vector<std::pair<Eigen::VectorXi, Eigen::VectorXi>> Configuration;

private:
    void clear();
    void AddConfigure(std::size_t, std::size_t, std::size_t, Layer::Types,
                      ActivationFunc::Types, LossFunc::Types, const Eigen::VectorXi &);

public:
    NeuralNetwork();
    ~NeuralNetwork();

    void AddInput(std::size_t, std::size_t);
    void AddConvolution(std::size_t, ActivationFunc::Types,
                        const Eigen::Matrix<int, 6, 1> &, std::size_t, std::size_t);
    void AddSub(Layer::Types, std::size_t, const Eigen::Vector4i &, std::size_t, std::size_t);
    void AddHidden(std::size_t, ActivationFunc::Types);
    void AddOutput(std::size_t, ActivationFunc::Types, LossFunc::Types);

    void InitialWeights(double);
    // возвращает набор пар (ошибка, номер итерации)
    DataSet<std::tuple<double, std::size_t>> Train(double,
                                                   const DataSet<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &,
                                                   std::size_t BatchSize);
    // возвращает набор кортежей (неверный индеск, посчитанное значение, ожидаемое значение)
    DataSet<std::tuple<int, int, int>> Fit(const DataSet<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &);
    int Reap(const Eigen::MatrixXd &);

    void WriteToFile(const char *);
    void ReadFromFile(const char *);
};
}// namespace KNet

#endif// NEURALNETWORK2_H
