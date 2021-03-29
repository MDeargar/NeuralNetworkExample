#ifndef LAYER2_H
#define LAYER2_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "activationfunc2.h"
#include "eigen_matrix_io.h"
#include "general.h"
#include <bits/stdc++.h>

namespace KNet {
class LossFunc;
class ActivationFunc;

class Layer {
public:
    enum Types {
        CONVOLUTION,
        MAXPOOLING,
        AVERAGEPOOLING,
        FULL
    };

public:
    Eigen::MatrixXd InputSignals;
    Eigen::MatrixXd InputSignalsProcessed;

    std::vector<Layer *> InputLayers;
    std::vector<Layer *> OutputLayers;

    ActivationFunc *Activate;

    virtual Eigen::MatrixXd BackPropogateGradient(Layer *) = 0;

public:
    Layer();
    Layer(std::size_t, std::size_t);
    virtual ~Layer() = 0;

    virtual void ReceiveSignal() = 0;
    virtual void CalculateGradient() = 0;

    virtual void CalculateWeightsDeltas() = 0;
    virtual void UpdateWeights(double) = 0;
    virtual void ResetUpdateHistory() = 0;

    virtual void InitialWeights(double) = 0;

    void SetInputValue(const Eigen::MatrixXd &);
    void SetActivationFunc(ActivationFunc *);

    virtual void WriteToFile(std::ofstream &) = 0;
    virtual void ReadFromFile(std::ifstream &) = 0;
};

class ConvolutionLayer : public Layer {
public:
    Eigen::Matrix<double, 1, 1> Bias;
    Eigen::Matrix<double, 1, 1> BiasDeltas;
    std::vector<Eigen::MatrixXd> Weights;
    std::vector<Eigen::MatrixXd> WeightsDeltas;
    // KerRow, KerCol, StepRow, StepCol, PaddingRow, PaddingCol
    std::vector<Eigen::Matrix<int, 6, 1>> KernelParameters;
    Eigen::MatrixXd Gradient;

public:
    Eigen::MatrixXd BackPropogateGradient(Layer *) override;

public:
    ConvolutionLayer();
    ConvolutionLayer(std::size_t, std::size_t);

    void ReceiveSignal() override;
    void CalculateGradient() override;

    void CalculateWeightsDeltas() override;
    void UpdateWeights(double) override;
    void ResetUpdateHistory() override;

    void InitialWeights(double) override;

    void WriteToFile(std::ofstream &) override;
    void ReadFromFile(std::ifstream &) override;
};

class MaxPooling : public Layer {
public:
    Eigen::MatrixXd Gradient;
    Eigen::Array<Eigen::Vector2i, Eigen::Dynamic, Eigen::Dynamic> ChosenIndexes;
    // KerRow, KerCol, StepRow, StepCol
    Eigen::Vector4i KernelParameters;

public:
    Eigen::MatrixXd BackPropogateGradient(Layer *) override;

public:
    MaxPooling();
    MaxPooling(std::size_t, std::size_t);

    void ReceiveSignal() override;
    void CalculateGradient() override;

    void CalculateWeightsDeltas() override;
    void UpdateWeights(double) override;
    void ResetUpdateHistory() override;

    void InitialWeights(double) override;

    void WriteToFile(std::ofstream &) override;
    void ReadFromFile(std::ifstream &) override;
};

class AveragePooling : public Layer {
public:
    Eigen::MatrixXd Gradient;
    // KerRow, KerCol, StepRow, StepCol
    Eigen::Vector4i KernelParameters;

public:
    Eigen::MatrixXd BackPropogateGradient(Layer *) override;

public:
    AveragePooling();
    AveragePooling(std::size_t, std::size_t);

    void ReceiveSignal() override;
    void CalculateGradient() override;

    void CalculateWeightsDeltas() override;
    void UpdateWeights(double) override;
    void ResetUpdateHistory() override;

    void InitialWeights(double) override;

    void WriteToFile(std::ofstream &) override;
    void ReadFromFile(std::ifstream &) override;
};

class FullLayer : public Layer {
public:
    LossFunc *LF;
    Eigen::VectorXd Gradient;

    Eigen::VectorXd Biases;
    Eigen::VectorXd BiasesDeltas;

    std::vector<Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1>> Weights;
    std::vector<Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1>> WeightsDeltas;

public:
    Eigen::MatrixXd BackPropogateGradient(Layer *) override;

public:
    FullLayer();
    FullLayer(std::size_t);

    void SetLossFunc(LossFunc &);
    void SetExpected(const Eigen::VectorXd &);
    void ReceiveSignal() override;
    void CalculateGradient() override;

    void CalculateWeightsDeltas() override;
    void UpdateWeights(double) override;
    void ResetUpdateHistory() override;

    void InitialWeights(double) override;

    void WriteToFile(std::ofstream &) override;
    void ReadFromFile(std::ifstream &) override;
};

void Connect(Layer &, ConvolutionLayer &, const Eigen::Matrix<int, 6, 1> &);
void Connect(Layer &, FullLayer &);
void Connect(ConvolutionLayer &, MaxPooling &, const Eigen::Vector4i &);
void Connect(ConvolutionLayer &, AveragePooling &, const Eigen::Vector4i &);

double CalculateLayerError(FullLayer &, const Eigen::VectorXd &);
int GetLayerResult(FullLayer &);
}// namespace KNet

#endif// LAYER2_H
