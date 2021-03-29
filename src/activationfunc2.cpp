#include "../include/activationfunc2.h"

namespace KNet {
const double HyperBolistic::A = 1;
const double HyperBolistic::B = 1;

ActivationFunc::~ActivationFunc() {
}

double HyperBolistic::Process(double value) {
    return A * tanh(B * value);
}

double HyperBolistic::Deprocess(double value) {
    return A * B * (1.0 - pow(tanh(B * value), 2.0));
}

double HyperBolistic::DeprocessOptimized(double value) {
    return B / A * (A - value) * (A + value);
}

Eigen::MatrixXd HyperBolistic::Process(const Eigen::MatrixXd &Mat) {
    return Mat.unaryExpr<double (*)(double)>(&Process);
}

Eigen::MatrixXd HyperBolistic::Deprocess(const Eigen::MatrixXd &Mat) {
    return Mat.unaryExpr<double (*)(double)>(&Deprocess);
}

Eigen::MatrixXd HyperBolistic::DeprocessOptimized(const Eigen::MatrixXd &Mat) {
    return Mat.unaryExpr<double (*)(double)>(&DeprocessOptimized);
}

Eigen::MatrixXd HyperBolistic::FullDerivative(const Eigen::VectorXd &V) {
    Eigen::MatrixXd Result(V.size(), V.size());
    Result.setZero();

    for (Eigen::Index i = 0; i < V.size(); i++) {
        Result(i, i) = DeprocessOptimized(V(i));
    }

    return Result;
}

Eigen::MatrixXd SoftMax::Process(const Eigen::MatrixXd &Mat) {
    Eigen::MatrixXd M = Mat.unaryExpr<double (*)(double)>(std::exp);
    return M / M.sum();
}

Eigen::MatrixXd SoftMax::Deprocess(const Eigen::MatrixXd &Mat) {
    Eigen::MatrixXd M = Mat.unaryExpr<double (*)(double)>(std::exp);
    M /= M.sum();

    for (Eigen::Index i = 0; i < M.rows(); i++) {
        for (Eigen::Index j = 0; j < M.cols(); j++) {
            M(i, j) *= 1.0 - M(i, j);
        }
    }

    return M;
}

Eigen::MatrixXd Identical::Process(const Eigen::MatrixXd &M) {
    return M;
}

Eigen::MatrixXd Identical::Deprocess(const Eigen::MatrixXd &M) {
    return Eigen::MatrixXd::Constant(M.rows(), M.cols(), 1);
}

Eigen::MatrixXd Identical::DeprocessOptimized(const Eigen::MatrixXd &M) {
    return Eigen::MatrixXd::Constant(M.rows(), M.cols(), 1);
}

Eigen::MatrixXd Identical::FullDerivative(const Eigen::VectorXd &M) {
    Eigen::MatrixXd Result(M.size(), M.size());

    for (Eigen::Index i = 0; i < M.size(); i++) {
        Result(i, i) = 1;
    }

    return Result;
}

Eigen::MatrixXd SoftMax::DeprocessOptimized(const Eigen::MatrixXd &M) {
    Eigen::MatrixXd Result(M.rows(), M.cols());
    Result = M;

    for (Eigen::Index i = 0; i < Result.rows(); i++) {
        for (Eigen::Index j = 0; j < Result.cols(); j++) {
            Result(i, j) *= 1.0 - Result(i, j);
        }
    }

    return Result;
}

Eigen::MatrixXd SoftMax::FullDerivative(const Eigen::VectorXd &V) {
    Eigen::MatrixXd Result(V.size(), V.size());

    for (Eigen::Index i = 0; i < V.size(); i++) {
        for (Eigen::Index j = 0; j < V.size(); j++) {
            Result(i, j) = (i == j) ? V(i) * (1 - V(i))
                                    : -V(i) * V(j);
        }
    }

    return Result;
}

LossFunc::~LossFunc() {
}

double ErrorSquare::CalculateError(const Eigen::VectorXd &Expected,
                                   const Eigen::VectorXd &Calculated)
{
    return 0.5 * (Expected - Calculated).squaredNorm();
}

Eigen::MatrixXd ErrorSquare::CalculateDerivative(const Eigen::VectorXd &Expected,
                                                 const Eigen::VectorXd &Calculated)
{
    return Expected - Calculated;
}

double CrossEntropy::CalculateError(const Eigen::VectorXd &Expected,
                                    const Eigen::VectorXd &Calculated)
{
    return -(Expected.array() * Calculated.unaryExpr<double (*)(double)>(std::log).array()).sum();
}

Eigen::MatrixXd CrossEntropy::CalculateDerivative(const Eigen::VectorXd &Expected,
                                                  const Eigen::VectorXd &Calculated)
{
    return -(Expected.array() / Calculated.array()).matrix();
}

ActivationFunc *HyperBolisticCreator::Produce() {
    static HyperBolistic Type;
    return &Type;
}

ActivationFunc *SoftMaxCreator::Produce() {
    static SoftMax Result;
    return &Result;
}

ActivationFunc *IdenticalCreator::Produce() {
    static Identical Result;
    return &Result;
}

LossFunc *ErrorSquareCreator::Produce() {
    static ErrorSquare Result;
    return &Result;
}

LossFunc *CrossEntropyCreator::Produce() {
    static CrossEntropy Result;
    return &Result;
}
}// namespace KNet
