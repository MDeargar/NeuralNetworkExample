#include "../include/layer2.h"

namespace KNet {
Layer::Layer() {
}

Layer::Layer(std::size_t Rows, std::size_t Cols)
    : InputSignals(Rows, Cols),
      InputSignalsProcessed(Rows, Cols),
      Activate(nullptr) {
    InputSignals.setZero();
    InputSignalsProcessed.setZero();
}

Layer::~Layer() {
}

void Layer::SetInputValue(const Eigen::MatrixXd &Value) {
    if (Value.rows() != InputSignals.rows() || Value.cols() != InputSignals.cols()) {
        throw generate_exception("Invalid number of rows or cols", __LINE__, __FILE__);
    }

    InputSignals = Value;
    InputSignalsProcessed = Activate->Process(InputSignals);
}

void Layer::SetActivationFunc(ActivationFunc *F) {
    Activate = F;
}

ConvolutionLayer::ConvolutionLayer() {
}

ConvolutionLayer::ConvolutionLayer(std::size_t Rows, std::size_t Cols)
    : Layer(Rows, Cols),
      Bias(),
      BiasDeltas(),
      Gradient(Rows, Cols) {
    Gradient.setZero();
}

void ConvolutionLayer::ReceiveSignal() {
    InputSignals.setZero();

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.cols(); j++) {
            for (std::size_t k = 0; k < InputLayers.size(); k++) {
                Layer &PrevL = *InputLayers[k];
                Eigen::Matrix<int, 6, 1> &Params = KernelParameters[k];
                Eigen::Index maxA = PrevL.InputSignals.rows() - i * Params(2) - 1 + Params(4);
                Eigen::Index maxB = PrevL.InputSignals.cols() - j * Params(3) - 1 + Params(5);
                Eigen::Index minA = Params(4) - i * Params(2);
                Eigen::Index minB = Params(5) - j * Params(3);

                if (Params(0) < maxA + 1) {
                    maxA = Params(0) - 1;
                }
                if (minA < 0) {
                    minA = 0;
                }


                if (Params(1) < maxB + 1) {
                    maxB = Params(1) - 1;
                }
                if (minB < 0) {
                    minB = 0;
                }

                InputSignals(i, j) += (Weights[k].block(minA, minB, maxA - minA + 1, maxB - minB + 1).array() *
                                       PrevL.InputSignalsProcessed.block(i * Params(2) + minA - Params(4),
                                                                         j * Params(3) + minB - Params(5),
                                                                         maxA - minA + 1,
                                                                         maxB - minB + 1).array()).sum();
            }

            InputSignals(i, j) += Bias(0, 0);
        }
    }

    InputSignalsProcessed = Activate->Process(InputSignals);
}

void ConvolutionLayer::CalculateGradient() {
    Gradient.setZero();

    for (std::size_t i = 0; i < OutputLayers.size(); i++) {
        Gradient += OutputLayers[i]->BackPropogateGradient(this);
    }
}

void ConvolutionLayer::CalculateWeightsDeltas() {
    Eigen::MatrixXd deprocessedOutputs = Activate->DeprocessOptimized(InputSignalsProcessed);

    for (std::size_t i = 0; i < InputLayers.size(); i++) {
        Layer &PrevL = *InputLayers[i];
        Eigen::Index KerRow = KernelParameters[i](0);
        Eigen::Index KerCol = KernelParameters[i](1);
        Eigen::Index StepRow = KernelParameters[i](2);
        Eigen::Index StepCol = KernelParameters[i](3);
        Eigen::Index PaddingRow = KernelParameters[i](4);
        Eigen::Index PaddingCol = KernelParameters[i](5);

        Eigen::MatrixXd WeightDelta(KerRow, KerCol);
        WeightDelta.setZero();

        for (Eigen::Index a = 0; a < KerRow; a++) {
            for (Eigen::Index b = 0; b < KerCol; b++) {
                Eigen::Index maxI = (PrevL.InputSignals.rows() - a + PaddingRow - 1) / StepRow;
                Eigen::Index maxJ = (PrevL.InputSignals.cols() - b + PaddingCol - 1) / StepCol;
                Eigen::Index minI = (PaddingRow - a) / StepRow;
                Eigen::Index minJ = (PaddingCol - b) / StepCol;

                if (InputSignals.rows() < maxI + 1) {
                    maxI = InputSignals.rows() - 1;
                }
                if (InputSignals.cols() < maxJ + 1) {
                    maxJ = InputSignals.cols() - 1;
                }
                if (minI < 0) {
                    minI = 0;
                }
                if (minJ < 0) {
                    minJ = 0;
                }

                /* Eigen::MatrixXd Corner = PrevL.InputSignalsProcessed.bottomRightCorner(PrevL.InputSignals.rows() - a + PaddingRow,
                                                                                        PrevL.InputSignals.cols() - b + PaddingCol);
                */
                Eigen::MatrixXd Corner = PrevL.InputSignalsProcessed.block(minI * StepRow + a - PaddingRow,
                                                                           minJ * StepCol + b - PaddingCol,
                                                                           StepRow * (maxI - minI) + 1,
                                                                           StepCol * (maxJ - minJ) + 1);
                Eigen::MatrixXd PrevOutputs = Eigen::Map<Eigen::MatrixXd, 0,
                                                         Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>(Corner.data(),
                                                                                                        maxI - minI + 1,
                                                                                                        maxJ - minJ + 1,
                                                                                                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                                                                                                Corner.outerStride() * StepRow,
                                                                                                                Corner.innerStride() * StepCol));
                WeightDelta(a, b) = (Gradient.block(minI, minJ, maxI - minI + 1, maxJ - minJ + 1).array() *
                                     PrevOutputs.array())
                                            .sum();
            }
        }

        WeightsDeltas[i] += WeightDelta;
    }

    BiasDeltas(0, 0) += Gradient.sum();
}

void ConvolutionLayer::UpdateWeights(double LearningRate) {
    for (std::size_t i = 0; i < Weights.size(); i++) {
        Weights[i] += LearningRate * WeightsDeltas[i];
    }
    Bias += LearningRate * BiasDeltas;
}

void ConvolutionLayer::ResetUpdateHistory() {
    for (std::size_t i = 0; i < WeightsDeltas.size(); i++) {
        WeightsDeltas[i].setZero();
    }
    BiasDeltas.setZero();
}

void ConvolutionLayer::InitialWeights(double Range) {
    if (Range < 0) {
        throw generate_exception("Range is less then zero", __LINE__, __FILE__);
    }

    Bias.setRandom();
    Bias *= Range;
    BiasDeltas.setZero();

    for (std::size_t i = 0; i < Weights.size(); i++) {
        Weights[i].setRandom();
        Weights[i] *= Range;
        WeightsDeltas[i].setZero();
    }
}

Eigen::MatrixXd ConvolutionLayer::BackPropogateGradient(Layer *PrevL) {
    Eigen::MatrixXd Result(PrevL->InputSignals.rows(), PrevL->InputSignals.cols());
    Result.setZero();

    std::vector<Layer *>::iterator FoundedIndex = std::find(InputLayers.begin(), InputLayers.end(), PrevL);
    if (FoundedIndex == InputLayers.end()) {
        throw generate_exception("Bad index", __LINE__, __FILE__);
    }
    Eigen::Index Ind = FoundedIndex - InputLayers.begin();
    Eigen::MatrixXd &WeightLink = Weights[Ind];
    Eigen::Index KerRow = KernelParameters[Ind](0);
    Eigen::Index KerCol = KernelParameters[Ind](1);
    Eigen::Index StepRow = KernelParameters[Ind](2);
    Eigen::Index StepCol = KernelParameters[Ind](3);
    Eigen::Index PaddingRow = KernelParameters[Ind](4);
    Eigen::Index PaddingCol = KernelParameters[Ind](5);

    Eigen::MatrixXd deprocessedPrevOutputs = PrevL->Activate->DeprocessOptimized(PrevL->InputSignalsProcessed);

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.cols(); j++) {
            Eigen::Index minRow = i * StepRow - PaddingRow;
            Eigen::Index maxRow = i * StepRow + KerRow - 1 - PaddingRow;
            Eigen::Index minCol = j * StepCol - PaddingCol;
            Eigen::Index maxCol = j * StepCol + KerCol - 1 - PaddingCol;

            if (maxRow + 1 > PrevL->InputSignals.rows()) {
                maxRow = PrevL->InputSignals.rows() - 1;
            }
            if (minRow < 0) {
                minRow = 0;
            }

            if (maxCol + 1 > PrevL->InputSignals.cols()) {
                maxCol = PrevL->InputSignals.cols() - 1;
            }
            if (minCol < 0) {
                minCol = 0;
            }

            Eigen::MatrixXd WMat = WeightLink.block(minRow - i * StepRow + PaddingRow,
                                                    minCol - j * StepCol + PaddingCol,
                                                    maxRow - minRow + 1,
                                                    maxCol - minCol + 1);
            Result.block(minRow, minCol, maxRow - minRow + 1, maxCol - minCol + 1) += Gradient(i, j) *
                                                                                      (WMat.array() *
                                                                                       deprocessedPrevOutputs.block(minRow,
                                                                                                                    minCol,
                                                                                                                    maxRow - minRow + 1,
                                                                                                                    maxCol - minCol + 1)
                                                                                                                    .array()).matrix();
        }
    }

    return Result;
}

void ConvolutionLayer::WriteToFile(std::ofstream &ostr) {
    Eigen::write_binary(ostr, Bias);

    for (std::size_t i = 0; i < Weights.size(); i++) {
        Eigen::write_binary(ostr, Weights[i]);
    }
}

void ConvolutionLayer::ReadFromFile(std::ifstream &istr) {
    Eigen::read_binary(istr, Bias);

    for (std::size_t i = 0; i < Weights.size(); i++) {
        Eigen::read_binary(istr, Weights[i]);
    }
}

MaxPooling::MaxPooling() {
}

MaxPooling::MaxPooling(std::size_t Rows, std::size_t Cols)
    : Layer(Rows, Cols),
      Gradient(Rows, Cols),
      ChosenIndexes(Rows, Cols) {
}

void MaxPooling::ReceiveSignal() {
    if (InputLayers.empty()) {
        throw generate_exception("No input layers!", __LINE__, __FILE__);
    }

    Layer *PrevL = InputLayers[0];
    Eigen::Index KerRow = KernelParameters(0);
    Eigen::Index KerCol = KernelParameters(1);
    Eigen::Index StepRow = KernelParameters(2);
    Eigen::Index StepCol = KernelParameters(3);

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.cols(); j++) {
            Eigen::Index ChosenRow;
            Eigen::Index ChosenCol;

            InputSignalsProcessed(i, j) = PrevL->InputSignalsProcessed.block(i * StepRow,
                                                                             j * StepCol,
                                                                             KerRow,
                                                                             KerCol).maxCoeff(&ChosenRow, &ChosenCol);
            ChosenIndexes(i, j) = Eigen::Vector2i(ChosenRow, ChosenCol);
            InputSignals(i, j) = PrevL->InputSignals(i * StepRow + ChosenRow, j * StepCol + ChosenCol);
        }
    }
}

void MaxPooling::CalculateGradient() {
    Gradient.setZero();

    for (Eigen::Index i = 0; i < OutputLayers.size(); i++) {
        Gradient += OutputLayers[i]->BackPropogateGradient(this);
    }
}

void MaxPooling::CalculateWeightsDeltas() {
}

void MaxPooling::UpdateWeights(double) {
}

void MaxPooling::ResetUpdateHistory() {
}

void MaxPooling::InitialWeights(double) {
}

Eigen::MatrixXd MaxPooling::BackPropogateGradient(Layer *L) {
    if (InputLayers.empty() || InputLayers[0] != L) {
        throw generate_exception("No Input layers or bad Layer", __LINE__, __FILE__);
    }

    Eigen::MatrixXd Result(L->InputSignals.rows(), L->InputSignals.cols());
    Result.setZero();

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.cols(); j++) {
            Result.block(i * KernelParameters(2),
                         j * KernelParameters(3),
                         KernelParameters(0),
                         KernelParameters(1)).operator()(ChosenIndexes(i, j)(0),
                                                                        ChosenIndexes(i, j)(1)) = Gradient(i, j);
        }
    }

    return Result;
}

void MaxPooling::WriteToFile(std::ofstream &) {
}

void MaxPooling::ReadFromFile(std::ifstream &) {
}

AveragePooling::AveragePooling() {
}

AveragePooling::AveragePooling(std::size_t Rows, std::size_t Cols)
    : Layer(Rows, Cols),
      Gradient(Rows, Cols) {
}

void AveragePooling::ReceiveSignal() {
    if (InputLayers.empty()) {
        throw generate_exception("No input layers!", __LINE__, __FILE__);
        ;
    }

    Layer *PrevL = InputLayers[0];

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.cols(); j++) {
            InputSignals(i, j) = PrevL->InputSignals.block(i * KernelParameters(2),
                                                           j * KernelParameters(3),
                                                           KernelParameters(0),
                                                           KernelParameters(1)).sum();
            InputSignalsProcessed(i, j) = PrevL->InputSignalsProcessed.block(i * KernelParameters(2),
                                                                             j * KernelParameters(3),
                                                                             KernelParameters(0),
                                                                             KernelParameters(1)).sum();
        }
    }

    InputSignals /= (KernelParameters(0) * KernelParameters(1));
    InputSignalsProcessed /= (KernelParameters(0) * KernelParameters(1));
}

void AveragePooling::CalculateGradient() {
    Gradient.setZero();

    for (std::size_t i = 0; i < OutputLayers.size(); i++) {
        Gradient += OutputLayers[i]->BackPropogateGradient(this);
    }
}

void AveragePooling::CalculateWeightsDeltas() {
}

void AveragePooling::UpdateWeights(double) {
}

void AveragePooling::ResetUpdateHistory() {
}

void AveragePooling::InitialWeights(double) {
}

Eigen::MatrixXd AveragePooling::BackPropogateGradient(Layer *PrevL) {
    if (InputLayers.empty() || InputLayers[0] != PrevL) {
        throw generate_exception("No Input layers or bad index!", __LINE__, __FILE__);
    }

    Eigen::MatrixXd Result(PrevL->InputSignals.rows(), PrevL->InputSignals.cols());
    Result.setZero();

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.cols(); j++) {
            Result.block(i * KernelParameters(2), j * KernelParameters(3),
                         KernelParameters(0), KernelParameters(1)).setConstant(Gradient(i, j));
        }
    }

    Result /= (KernelParameters(0) * KernelParameters(1));

    return Result;
}

void AveragePooling::WriteToFile(std::ofstream &) {
}

void AveragePooling::ReadFromFile(std::ifstream &) {
}

FullLayer::FullLayer() {
}

FullLayer::FullLayer(std::size_t Num)
    : Layer(Num, 1),
      Gradient(Num),
      Biases(Num),
      BiasesDeltas(Num) {
}

void FullLayer::SetLossFunc(LossFunc &F) {
    LF = &F;
}

void FullLayer::SetExpected(const Eigen::VectorXd &Expected) {
    if (Expected.rows() != InputSignals.rows() || Expected.cols() != InputSignals.cols()) {
        throw generate_exception("Bad num of cols or rows", __LINE__, __FILE__);
        ;
    }

    Eigen::VectorXd V = InputSignalsProcessed;
    Eigen::MatrixXd Res = (LF->CalculateDerivative(Expected, V).transpose() * Activate->FullDerivative(V)).transpose();

    if (Res.rows() != Gradient.rows() || Res.cols() != Gradient.cols()) {
        throw generate_exception("Bad num of cols or rows", __LINE__, __FILE__);
        ;
    }

    Gradient = Res;
}

void FullLayer::ReceiveSignal() {
    InputSignals.setZero();

    for (std::size_t i = 0; i < InputLayers.size(); i++) {
        Eigen::MatrixXd &InputData = InputLayers[i]->InputSignalsProcessed;

        for (Eigen::Index j = 0; j < InputSignals.rows(); j++) {
            Eigen::MatrixXd &WLink = Weights[i](j);
            InputSignals(j, 0) += (WLink.array() * InputData.array()).sum();
        }
    }

    InputSignals += Biases;
    InputSignalsProcessed = Activate->Process(InputSignals);
}

void FullLayer::CalculateGradient() {
    if (!OutputLayers.empty()) {
        Gradient.setZero();
    }

    for (std::size_t i = 0; i < OutputLayers.size(); i++) {
        Gradient += OutputLayers[i]->BackPropogateGradient(this);
    }
}

void FullLayer::CalculateWeightsDeltas() {
    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        for (std::size_t j = 0; j < InputLayers.size(); j++) {
            WeightsDeltas[j](i) += InputLayers[j]->InputSignalsProcessed * Gradient(i, 0);
        }
    }

    BiasesDeltas += Gradient;
}

void FullLayer::UpdateWeights(double LearningRete) {
    for (std::size_t i = 0; i < InputLayers.size(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.rows(); j++) {
            Weights[i](j) += WeightsDeltas[i](j) * LearningRete;
        }
    }

    Biases += BiasesDeltas * LearningRete;
}

void FullLayer::ResetUpdateHistory() {
    for (std::size_t i = 0; i < InputLayers.size(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.rows(); j++) {
            WeightsDeltas[i](j).setZero();
        }
    }

    BiasesDeltas.setZero();
}

void FullLayer::InitialWeights(double Range) {
    if (Range < 0) {
        throw generate_exception("Range is less then zero!", __LINE__, __FILE__);
    }

    Biases.setRandom();
    Biases *= Range;
    BiasesDeltas.setZero();

    for (std::size_t i = 0; i < Weights.size(); i++) {
        for (Eigen::Index j = 0; j < InputSignals.rows(); j++) {
            Weights[i](j).setRandom();
            Weights[i](j) *= Range;

            WeightsDeltas[i](j).setZero();
        }
    }
}
/* не уверен, что правильно считается градиент */
Eigen::MatrixXd FullLayer::BackPropogateGradient(Layer *PrevL) {
    std::vector<Layer *>::iterator FoundedIndex = std::find(InputLayers.begin(), InputLayers.end(), PrevL);
    if (FoundedIndex == InputLayers.end()) {
        throw generate_exception("Layer not found!", __LINE__, __FILE__);
        ;
    }
    Eigen::MatrixXd Result(PrevL->InputSignals.rows(), PrevL->InputSignals.cols());

    std::size_t Ind = FoundedIndex - InputLayers.begin();
    Eigen::MatrixXd &InputMat = InputLayers[Ind]->InputSignalsProcessed;
    Eigen::MatrixXd Grad(Result.rows(), Result.cols());
    Grad.setZero();

    for (Eigen::Index i = 0; i < InputSignals.rows(); i++) {
        Grad += Weights[Ind](i) * Gradient(i, 0);
    }

    Result = (Grad.array() * PrevL->Activate->DeprocessOptimized(InputMat).array()).matrix();

    return Result;
}

void FullLayer::WriteToFile(std::ofstream &ostr) {
    Eigen::write_binary(ostr, Biases);

    for (std::size_t i = 0; i < Weights.size(); i++) {
        for (std::size_t j = 0; j < Weights[i].size(); j++) {
            Eigen::write_binary(ostr, Weights[i](j, 0));
        }
    }
}

void FullLayer::ReadFromFile(std::ifstream &istr) {
    Eigen::read_binary(istr, Biases);

    for (std::size_t i = 0; i < Weights.size(); i++) {
        for (std::size_t j = 0; j < Weights[i].size(); j++) {
            Eigen::read_binary(istr, Weights[i](j, 0));
        }
    }
}

void Connect(Layer &PrevL, ConvolutionLayer &NextL, const Eigen::Matrix<int, 6, 1> &Params) {
    if (NextL.InputSignals.rows() != (PrevL.InputSignals.rows() + Params(4) * 2 - Params(0)) / Params(2) + 1) {
        throw generate_exception("Invalid size", __LINE__, __FILE__);
    } else if (NextL.InputSignals.cols() != (PrevL.InputSignals.cols() + Params(5) * 2 - Params(1)) / Params(3) + 1) {
        throw generate_exception("Invalid size", __LINE__, __FILE__);
    }

    NextL.Weights.push_back(Eigen::MatrixXd(Params(0), Params(1)));
    NextL.WeightsDeltas.push_back(Eigen::MatrixXd(Params(0), Params(1)));

    NextL.KernelParameters.push_back(Params);
    NextL.InputLayers.push_back(&PrevL);
    PrevL.OutputLayers.push_back(&NextL);
}

void Connect(Layer &PrevL, FullLayer &NextL) {
    NextL.InputLayers.push_back(&PrevL);
    PrevL.OutputLayers.push_back(&NextL);

    Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1> TempW(NextL.InputSignals.rows());
    for (Eigen::Index i = 0; i < NextL.InputSignals.rows(); i++) {
        TempW(i, 0) = Eigen::MatrixXd(PrevL.InputSignals.rows(), PrevL.InputSignals.cols());
    }

    NextL.Weights.push_back(TempW);
    NextL.WeightsDeltas.push_back(TempW);
}

void Connect(ConvolutionLayer &PrevL, MaxPooling &NextL, const Eigen::Vector4i &P) {
    if ((PrevL.InputSignals.rows() - P(0)) % P(2) != 0 ||
        (PrevL.InputSignals.cols() - P(1)) % P(3) != 0)
    {
        throw generate_exception("Bad arguments", __LINE__, __FILE__);
    } else if ((PrevL.InputSignals.rows() - P(0)) / P(2) + 1 != NextL.InputSignals.rows() ||
               (PrevL.InputSignals.cols() - P(1)) / P(3) + 1 != NextL.InputSignals.cols())
    {
        throw generate_exception("Bad arguments", __LINE__, __FILE__);
    }

    NextL.InputLayers.push_back(&PrevL);
    PrevL.OutputLayers.push_back(&NextL);
    NextL.KernelParameters = P;

    NextL.Activate = PrevL.Activate;
}

void Connect(ConvolutionLayer &PrevL, AveragePooling &NextL, const Eigen::Vector4i &P) {
    if ((PrevL.InputSignals.rows() - P(0)) % P(2) != 0 ||
        (PrevL.InputSignals.cols() - P(1)) % P(3) != 0)
    {
        throw generate_exception("Bad arguments", __LINE__, __FILE__);
    } else if ((PrevL.InputSignals.rows() - P(0)) / P(2) + 1 != NextL.InputSignals.rows() ||
               (PrevL.InputSignals.cols() - P(1)) / P(3) + 1 != NextL.InputSignals.cols())
    {
        throw generate_exception("Bad arguments", __LINE__, __FILE__);
    }

    NextL.InputLayers.push_back(&PrevL);
    PrevL.OutputLayers.push_back(&NextL);
    NextL.KernelParameters = P;

    NextL.Activate = PrevL.Activate;
}

double CalculateLayerError(FullLayer &L, const Eigen::VectorXd &Expected) {
    Eigen::VectorXd V = L.InputSignalsProcessed;
    return L.LF->CalculateError(Expected, V);
}

int GetLayerResult(FullLayer &L) {
    Eigen::Index i, j;
    L.InputSignalsProcessed.maxCoeff(&i, &j);
    return i;
}
}// namespace KNet
