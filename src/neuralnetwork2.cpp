#include "../include/neuralnetwork2.h"

namespace KNet {
NeuralNetwork::NeuralNetwork() {
}

NeuralNetwork::~NeuralNetwork() {
    clear();
}

void NeuralNetwork::AddConfigure(std::size_t features,
                                 std::size_t rows,
                                 std::size_t cols,
                                 Layer::Types LType,
                                 ActivationFunc::Types FType,
                                 LossFunc::Types LF,
                                 const Eigen::VectorXi &CParams)
{
    Eigen::VectorXi LParams(6);
    LParams << features, rows, cols, LType, FType, LF;
    Configuration.push_back(std::make_pair(LParams, CParams));
}

void NeuralNetwork::AddInput(std::size_t Rows, std::size_t Cols) {
    std::vector<Layer *> Vec;
    Vec.push_back(new ConvolutionLayer(Rows, Cols));
    Vec[0]->SetActivationFunc(IdenticalCreator::Produce());

    Stratums.push_back(std::make_pair(Vec, Layer::Types::CONVOLUTION));

    AddConfigure(1, Rows, Cols, Layer::Types::CONVOLUTION,
                 ActivationFunc::IDENTICAL, LossFunc::Types(-1),
                 Eigen::VectorXi());
}

void NeuralNetwork::AddConvolution(std::size_t Amount,
                                   ActivationFunc::Types ActType,
                                   const Eigen::Matrix<int, 6, 1> &Params,
                                   std::size_t Rows,
                                   std::size_t Cols)
{
    std::vector<Layer *> Vec;
    for (std::size_t i = 0; i < Amount; i++) {
        Vec.push_back(new ConvolutionLayer(Rows, Cols));
        switch (ActType) {
            case ActivationFunc::Types::HYPERBOLISTIC:
                Vec.back()->SetActivationFunc(HyperBolisticCreator::Produce());
                break;
            case ActivationFunc::Types::IDENTICAL:
                Vec.back()->SetActivationFunc(IdenticalCreator::Produce());
                break;
            case ActivationFunc::Types::SOFTMAX:
                Vec.back()->SetActivationFunc(SoftMaxCreator::Produce());
                break;
            default:
                throw generate_exception("Unexpected case", __LINE__, __FILE__);
                ;
        }
    }

    for (std::size_t i = 0; i < Vec.size(); i++) {
        for (std::size_t j = 0; j < Stratums.back().first.size(); j++) {
            Connect(*Stratums.back().first[j], *dynamic_cast<ConvolutionLayer *>(Vec[i]), Params);
        }
    }

    Stratums.push_back(std::make_pair(Vec, Layer::Types::CONVOLUTION));

    AddConfigure(Amount, Rows, Cols, Layer::Types::CONVOLUTION, ActType, LossFunc::Types(-1), Params);
}

void NeuralNetwork::AddSub(Layer::Types LType,
                           std::size_t Amount,
                           const Eigen::Vector4i &Params,
                           std::size_t KerRow,
                           std::size_t KerCol)
{
    if (Stratums.back().second != Layer::Types::CONVOLUTION) {
        throw generate_exception("Bad layer type", __LINE__, __FILE__);
    } else if (Amount != Stratums.back().first.size()) {
        throw generate_exception("Bad num of layers", __LINE__, __FILE__);
        ;
    }

    std::vector<Layer *> Vec;
    for (std::size_t i = 0; i < Amount; i++) {
        switch (LType) {
            case Layer::Types::MAXPOOLING:
                Vec.push_back(new MaxPooling(KerRow, KerCol));
                break;
            case Layer::Types::AVERAGEPOOLING:
                Vec.push_back(new AveragePooling(KerRow, KerCol));
                break;
            default:
                throw generate_exception("Unexpected case", __LINE__, __FILE__);
        }
    }

    for (std::size_t i = 0; i < Amount; i++) {
        switch (LType) {
            case Layer::Types::MAXPOOLING:
                Connect(*dynamic_cast<ConvolutionLayer *>(Stratums.back().first[i]), *dynamic_cast<MaxPooling *>(Vec[i]), Params);
                break;
            case Layer::Types::AVERAGEPOOLING:
                Connect(*dynamic_cast<ConvolutionLayer *>(Stratums.back().first[i]), *dynamic_cast<AveragePooling *>(Vec[i]), Params);
                break;
        }
    }

    Stratums.push_back(std::make_pair(Vec, LType));

    AddConfigure(Amount, KerRow, KerCol, LType, ActivationFunc::Types(-1), LossFunc::Types(-1), Params);
}

void NeuralNetwork::AddHidden(std::size_t LayerSize, ActivationFunc::Types ActType) {
    std::vector<Layer *> Vec;
    Vec.push_back(new FullLayer(LayerSize));

    switch (ActType) {
        case ActivationFunc::Types::SOFTMAX:
            Vec[0]->SetActivationFunc(SoftMaxCreator::Produce());
            break;
        case ActivationFunc::Types::IDENTICAL:
            Vec[0]->SetActivationFunc(IdenticalCreator::Produce());
            break;
        case ActivationFunc::Types::HYPERBOLISTIC:
            Vec[0]->SetActivationFunc(HyperBolisticCreator::Produce());
            break;
    }

    for (std::size_t i = 0; i < Stratums.back().first.size(); i++) {
        Connect(*Stratums.back().first[i], *dynamic_cast<FullLayer *>(Vec[0]));
    }

    Stratums.push_back(std::make_pair(Vec, Layer::Types::FULL));

    AddConfigure(LayerSize, 1, 1, Layer::Types::FULL, ActType, LossFunc::Types(-1), Eigen::VectorXi());
}

void NeuralNetwork::AddOutput(std::size_t LayerSize, ActivationFunc::Types ActType, LossFunc::Types LF) {
    std::vector<Layer *> Vec;
    Vec.push_back(new FullLayer(LayerSize));

    switch (ActType) {
        case ActivationFunc::Types::SOFTMAX:
            Vec[0]->SetActivationFunc(SoftMaxCreator::Produce());
            break;
        case ActivationFunc::Types::IDENTICAL:
            Vec[0]->SetActivationFunc(IdenticalCreator::Produce());
            break;
        case ActivationFunc::Types::HYPERBOLISTIC:
            Vec[0]->SetActivationFunc(HyperBolisticCreator::Produce());
            break;
    }

    switch (LF) {
        case LossFunc::Types::ERRORSQUARE:
            dynamic_cast<FullLayer *>(Vec[0])->SetLossFunc(*ErrorSquareCreator::Produce());
            break;
        case LossFunc::Types::CROSSENTROPY:
            dynamic_cast<FullLayer *>(Vec[0])->SetLossFunc(*CrossEntropyCreator::Produce());
            break;
        default:
            throw generate_exception("Unexpected case", __LINE__, __FILE__);
    }

    for (std::size_t i = 0; i < Stratums.back().first.size(); i++) {
        Connect(*Stratums.back().first[i], *dynamic_cast<FullLayer *>(Vec[0]));
    }

    Stratums.push_back(std::make_pair(Vec, Layer::Types::FULL));

    AddConfigure(LayerSize, 1, 1, Layer::Types::FULL, ActType, LF, Eigen::VectorXi());
}

void NeuralNetwork::InitialWeights(double Range) {
    for (std::size_t i = 0; i < Stratums.size(); i++) {
        for (std::size_t j = 0; j < Stratums[i].first.size(); j++) {
            Stratums[i].first[j]->InitialWeights(Range);
        }
    }
}

DataSet<std::tuple<double, std::size_t>> NeuralNetwork::Train(double LearningRate,
                                                              const DataSet<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &Data,
                                                              std::size_t BatchSize)
{
    std::size_t BatchIndex = 0;
    double BatchError = 0;
    DataSet<std::tuple<double, std::size_t>> result;

    for (std::size_t i = 0; i < Data.Size(); i++, ++BatchIndex) {
        if (BatchIndex == BatchSize) {
            BatchIndex = 0;

            for (std::size_t j = 0; j < Stratums.size(); j++) {
                for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
                    Stratums[j].first[k]->UpdateWeights(LearningRate);
                    Stratums[j].first[k]->ResetUpdateHistory();
                }
            }

            result.Push(std::make_tuple(BatchError, i));

            BatchError = 0;
        }

        Stratums[0].first[0]->SetInputValue(Data[i].first);
        for (std::size_t j = 1; j < Stratums.size(); j++) {
            for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
                Stratums[j].first[k]->ReceiveSignal();
            }
        }

        FullLayer *OutputLayer = dynamic_cast<FullLayer *>(Stratums.back().first[0]);
        OutputLayer->SetExpected(Data[i].second);
        BatchError += CalculateLayerError(*OutputLayer, Data[i].second);

        for (std::size_t j = Stratums.size() - 1; j >= 1; j--) {
            for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
                Stratums[j].first[k]->CalculateGradient();
            }
        }

        for (std::size_t j = 1; j < Stratums.size(); j++) {
            for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
                Stratums[j].first[k]->CalculateWeightsDeltas();
            }
        }
    }

    for (std::size_t j = 0; j < Stratums.size(); j++) {
        for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
            Stratums[j].first[k]->ResetUpdateHistory();
        }
    }

    return result;
}

DataSet<std::tuple<int, int, int>> NeuralNetwork::Fit(const DataSet<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &Data) {
    int errCount = 0;
    int nnAns;
    DataSet<std::tuple<int, int, int>> result;

    for (std::size_t i = 0; i < Data.Size(); i++) {
        int correctAns, temp;
        Data[i].second.maxCoeff(&correctAns, &temp);

        Stratums[0].first[0]->SetInputValue(Data[i].first);
        for (std::size_t j = 1; j < Stratums.size(); j++) {
            for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
                Stratums[j].first[k]->ReceiveSignal();
            }
        }

        FullLayer *OutputLayer = dynamic_cast<FullLayer *>(Stratums.back().first[0]);

        if (correctAns != (nnAns = GetLayerResult(*OutputLayer))) {
            ++errCount;
            result.Push(std::make_tuple(i, nnAns, correctAns));
        }
    }

    for (std::size_t j = 0; j < Stratums.size(); j++) {
        for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
            Stratums[j].first[k]->ResetUpdateHistory();
        }
    }

    return result;
}

int NeuralNetwork::Reap(const Eigen::MatrixXd &M) {
    Stratums[0].first[0]->SetInputValue(M);
    for (std::size_t j = 1; j < Stratums.size(); j++) {
        for (std::size_t k = 0; k < Stratums[j].first.size(); k++) {
            Stratums[j].first[k]->ReceiveSignal();
        }
    }

    return GetLayerResult(*dynamic_cast<FullLayer *>(Stratums.back().first[0]));
}

void NeuralNetwork::WriteToFile(const char *filename) {
    std::ofstream ostr(filename, std::ios_base::binary | std::ios_base::trunc | std::ios_base::out);
    if (!ostr.is_open()) {
        throw generate_exception("Couldn't open file", __LINE__, __FILE__);
        ;
    }

    std::size_t sz = Stratums.size();
    ostr.write(reinterpret_cast<const char *>(&sz), sizeof(sz));

    for (std::size_t i = 0; i < Stratums.size(); i++) {
        Eigen::write_binary(ostr, Configuration[i].first);
        Eigen::write_binary(ostr, Configuration[i].second);

        for (std::size_t j = 0; j < Stratums[i].first.size(); j++) {
            Stratums[i].first[j]->WriteToFile(ostr);
        }
    }
}

void NeuralNetwork::ReadFromFile(const char *filename) {
    std::ifstream istr(filename, std::ios_base::in | std::ios_base::binary);
    if (!istr.is_open()) {
        throw generate_exception("Couldn't open file", __LINE__, __FILE__);
    }

    clear();
    std::size_t sz;
    istr.read(reinterpret_cast<char *>(&sz), sizeof(sz));

    for (std::size_t i = 0; i < sz; i++) {
        Eigen::VectorXi LParams;
        Eigen::VectorXi ConnParams;

        Eigen::read_binary(istr, LParams);
        Eigen::read_binary(istr, ConnParams);

        std::size_t features = LParams(0);
        std::size_t rows = LParams(1);
        std::size_t cols = LParams(2);
        Layer::Types LType = Layer::Types(LParams(3));
        ActivationFunc::Types FType = ActivationFunc::Types(LParams(4));

        if (i == 0) {
            AddInput(rows, cols);
        } else if (LType == Layer::CONVOLUTION) {
            AddConvolution(features, FType, ConnParams, rows, cols);
        } else if (LType == Layer::MAXPOOLING || LType == Layer::AVERAGEPOOLING) {
            AddSub(LType, features, ConnParams, rows, cols);
        } else if (LType == Layer::Types::FULL && i + 1 != sz) {
            AddHidden(features, FType);
        } else if (LType == Layer::Types::FULL) {
            AddOutput(features, FType, LossFunc::Types(LParams(5)));
        }

        for (std::size_t j = 0; j < Stratums.back().first.size(); j++) {
            Stratums.back().first[j]->ReadFromFile(istr);
        }
    }
}

void NeuralNetwork::clear() {
    for (std::size_t i = 0; i < Stratums.size(); i++) {
        for (std::size_t j = 0; j < Stratums[i].first.size(); j++) {
            delete Stratums[i].first[j];
        }
    }

    Stratums.clear();
    Configuration.clear();
}
}// namespace KNet
