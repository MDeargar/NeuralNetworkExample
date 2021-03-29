#ifndef ACTIVATIONFUNC2_H
#define ACTIVATIONFUNC2_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "general.h"

namespace KNet {
class ActivationFunc {
public:
    enum Types {
        HYPERBOLISTIC,
        IDENTICAL,
        SOFTMAX
    };

public:
    virtual ~ActivationFunc() = 0;

    virtual Eigen::MatrixXd Process(const Eigen::MatrixXd &) = 0;
    virtual Eigen::MatrixXd Deprocess(const Eigen::MatrixXd &) = 0;
    virtual Eigen::MatrixXd DeprocessOptimized(const Eigen::MatrixXd &) = 0;

    virtual Eigen::MatrixXd FullDerivative(const Eigen::VectorXd &) = 0;
};

class HyperBolistic : public ActivationFunc {
    static const double A;
    static const double B;

private:
    static double Process(double);
    static double Deprocess(double);
    static double DeprocessOptimized(double);

public:
    Eigen::MatrixXd Process(const Eigen::MatrixXd &) override;
    Eigen::MatrixXd Deprocess(const Eigen::MatrixXd &) override;
    Eigen::MatrixXd DeprocessOptimized(const Eigen::MatrixXd &) override;

    Eigen::MatrixXd FullDerivative(const Eigen::VectorXd &) override;
};

class SoftMax : public ActivationFunc {
public:
    Eigen::MatrixXd Process(const Eigen::MatrixXd &) override;
    Eigen::MatrixXd Deprocess(const Eigen::MatrixXd &) override;
    Eigen::MatrixXd DeprocessOptimized(const Eigen::MatrixXd &) override;

    Eigen::MatrixXd FullDerivative(const Eigen::VectorXd &) override;
};

class Identical : public ActivationFunc {
public:
    Eigen::MatrixXd Process(const Eigen::MatrixXd &) override;
    Eigen::MatrixXd Deprocess(const Eigen::MatrixXd &) override;
    Eigen::MatrixXd DeprocessOptimized(const Eigen::MatrixXd &) override;

    Eigen::MatrixXd FullDerivative(const Eigen::VectorXd &) override;
};

class LossFunc {
public:
    enum Types {
        ERRORSQUARE,
        CROSSENTROPY
    };

public:
    virtual ~LossFunc() = 0;

    // Expected, Calculated
    virtual double CalculateError(const Eigen::VectorXd &, const Eigen::VectorXd &) = 0;
    virtual Eigen::MatrixXd CalculateDerivative(const Eigen::VectorXd &, const Eigen::VectorXd &) = 0;
};

class ErrorSquare : public LossFunc {
public:
    double CalculateError(const Eigen::VectorXd &, const Eigen::VectorXd &) override;
    Eigen::MatrixXd CalculateDerivative(const Eigen::VectorXd &, const Eigen::VectorXd &) override;
};

class CrossEntropy : public LossFunc {
public:
    double CalculateError(const Eigen::VectorXd &, const Eigen::VectorXd &) override;
    Eigen::MatrixXd CalculateDerivative(const Eigen::VectorXd &, const Eigen::VectorXd &) override;
};

class HyperBolisticCreator {
public:
    HyperBolisticCreator() = delete;

    static ActivationFunc *Produce();
};

class SoftMaxCreator {
public:
    SoftMaxCreator() = delete;

    static ActivationFunc *Produce();
};

class IdenticalCreator {
public:
    IdenticalCreator() = delete;

    static ActivationFunc *Produce();
};

class ErrorSquareCreator {
public:
    ErrorSquareCreator() = delete;

    static LossFunc *Produce();
};

class CrossEntropyCreator {
public:
    CrossEntropyCreator() = delete;

    static LossFunc *Produce();
};
}// namespace KNet

#endif// ACTIVATIONFUNC2_H
