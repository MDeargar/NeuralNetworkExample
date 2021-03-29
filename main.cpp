#include "include/Eigen/Core"
#include "include/Eigen/Dense"
#include "include/activationfunc2.h"
#include "include/dataset.h"
#include "include/eigen_matrix_io.h"
#include "include/layer2.h"
#include "include/neuralnetwork2.h"
#include <bits/stdc++.h>

using namespace Eigen;
using KNet::generate_exception;

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void ReadMNISTData(std::vector<Eigen::MatrixXd> &Result, const char *filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char *) &number_of_images, sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char *) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char *) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        Result.resize(number_of_images);

        for (int i = 0; i < number_of_images; ++i) {
            Result[i] = Eigen::MatrixXd(n_rows, n_cols);

            for (int r = 0; r < n_rows; ++r) {
                for (int c = 0; c < n_cols; ++c) {
                    unsigned char temp = 0;
                    file.read((char *) &temp, sizeof(temp));
                    Result[i](r, c) = (double) temp;
                }
            }
        }
    } else {
        throw generate_exception(std::string("Can't open file ") + filename, __LINE__, __FILE__);
    }
}

void ReadMnistTrainLabel(std::vector<VectorXd> &arr, const char *filename) {
    std::ifstream file(filename, std::ios_base::binary | std::ios_base::in);

    if (!file.is_open()) {
        throw generate_exception(std::string("Can't open file ") + filename, __LINE__, __FILE__);
    }

    int magic_number, number_of_images;
    magic_number = number_of_images = 0;

    file.read((char *) &magic_number, sizeof(magic_number));
    file.read((char *) &number_of_images, sizeof(number_of_images));
    number_of_images = ReverseInt(number_of_images);

    arr.resize(number_of_images);

    for (int i = 0; i < number_of_images; i++) {
        arr[i] = VectorXd(10);

        for (int j = 0; j < 10; j++)
            arr[i](j) = 0.0;

        unsigned char temp = 0;
        file.read((char *) &temp, sizeof(temp));
        arr[i](temp) = 1.0;
    }

    file.close();
}

double NormFunc(double x) {
    return x / 255.0;
}

void NormalizeInputs(const std::vector<MatrixXd> &In,
                     std::vector<MatrixXd> &Out,
                     std::size_t AddRow,
                     std::size_t AddCol) {
    if (In.size() != 60000 && In.size() != 10000) {
        throw generate_exception("Unexpected train data size", __LINE__, __FILE__);
    }

    Out.resize(In.size());
    for (std::size_t i = 0; i < In.size(); i++) {
        if (In[i].rows() != 28 || In[i].cols() != 28) {
            throw generate_exception("Unexpected train data size", __LINE__, __FILE__);
        }
        Out[i] = MatrixXd::Constant(In[i].rows() + AddRow * 2, In[i].cols() + AddCol * 2, 0.0);
        Out[i].block(AddRow, AddCol, 28, 28) = In[i];
        Out[i] = Out[i].unaryExpr<double (*)(double)>(&NormFunc);
    }
}

void MixTrainData(std::vector<MatrixXd> &Input, std::vector<VectorXd> &Output) {
    std::vector<int> Indexes;
    std::vector<int> Temp(Input.size());

    for (int i = 0; i < Input.size(); i++) {
        Temp[i] = i;
    }

    while (Indexes.size() < Input.size()) {
        int ind = rand() % Temp.size();
        Indexes.push_back(Temp[ind]);
        Temp.erase(Temp.begin() + ind);
    }

    std::vector<MatrixXd> InputResult;
    std::vector<VectorXd> OutputResult;

    for (std::size_t i = 0; i < Indexes.size(); i++) {
        InputResult.push_back(Input[Indexes[i]]);
        OutputResult.push_back(Output[Indexes[i]]);
    }

    Input = InputResult;
    Output = OutputResult;
}

void InitialCNN(KNet::NeuralNetwork &Net) {
    Net.AddInput(28, 28);
    Eigen::Matrix<int, 6, 1> KerPar;
    Eigen::Vector4i SubPar;

    KerPar << 5, 5, 1, 1, 2, 2;
    Net.AddConvolution(6, KNet::ActivationFunc::Types::HYPERBOLISTIC, KerPar, 28, 28);

    SubPar << 2, 2, 2, 2;
    Net.AddSub(KNet::Layer::Types::MAXPOOLING, 6, SubPar, 14, 14);

    KerPar << 5, 5, 1, 1, 0, 0;
    Net.AddConvolution(16, KNet::ActivationFunc::Types::HYPERBOLISTIC, KerPar, 10, 10);

    SubPar << 2, 2, 2, 2;
    Net.AddSub(KNet::Layer::Types::MAXPOOLING, 16, SubPar, 5, 5);

    Net.AddHidden(120, KNet::ActivationFunc::HYPERBOLISTIC);
    Net.AddHidden(84, KNet::ActivationFunc::HYPERBOLISTIC);
    Net.AddOutput(10, KNet::ActivationFunc::SOFTMAX, KNet::LossFunc::ERRORSQUARE);

    Net.InitialWeights(0.05);
}

void TrainFunc(KNet::NeuralNetwork &CNN, KNet::DataSet<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> &Data) {
    CNN.Train(0.01, Data, 128);
}

template<typename T>
std::string PutImage(T &&M) {
    std::ostringstream ostr;

    for (Eigen::Index i = 0; i < M.rows(); i++) {
        for (Eigen::Index j = 0; j + 1 < M.cols(); j++) {
            ostr << M(i, j) << '\t';
        }
        ostr << M(i, M.cols() - 1);

        if (i + 1 != M.rows()) {
            ostr << '\n';
        }
    }

    return ostr.str();
}

int main(int argc, char *argv[]) {
    try {
        srand(time(0));

        if (argc != 7) {
            throw generate_exception("Bad arguments in main function", __LINE__, __FILE__);
            ;
        }

        const char *filename1 = argv[1];
        const char *filename2 = argv[2];

        std::vector<MatrixXd> InputTemp;
        std::vector<MatrixXd> InputData;
        std::vector<VectorXd> OutputData;
        ReadMNISTData(InputTemp, argv[3]);
        ReadMnistTrainLabel(OutputData, argv[4]);
        NormalizeInputs(InputTemp, InputData, 0, 0);

        std::vector<MatrixXd> InputTempCheck;
        std::vector<MatrixXd> InputDataCheck;
        std::vector<VectorXd> OutputDataCheck;
        ReadMNISTData(InputTempCheck, argv[5]);
        ReadMnistTrainLabel(OutputDataCheck, argv[6]);
        NormalizeInputs(InputTempCheck, InputDataCheck, 0, 0);

        KNet::DataSet<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> dataTrain, dataTest;
        for (int i = 0; i < InputData.size(); i++) {
            dataTrain.Push({InputData[i], OutputData[i]});
        }
        for (int i = 0; i < InputDataCheck.size(); i++) {
            dataTest.Push({InputDataCheck[i], OutputDataCheck[i]});
        }

        dataTrain.MixData();
//        dataTest.MixData();

        KNet::NeuralNetwork CNN;
        InitialCNN(CNN);
        CNN.ReadFromFile(filename2);
        /*
		for(int i = 0; i < 10; i++) {
	   		TrainFunc(CNN, dataTrain);
	    }
	    */
        auto x = CNN.Fit(dataTest);
        std::cout << "Accuracy is " << double(dataTest.Size() - x.Size()) / dataTest.Size() * 100.0 << std::endl;
        std::cout << "Invalid tuples are:\n";
        for (int i = 0; i < x.Size(); i++) {
            std::cout << std::get<0>(x[i]) << ' ' << std::get<1>(x[i]) << ' ' << std::get<2>(x[i]) << std::endl;
        }
    } catch (std::exception &ex) {
        std::cout << ex.what() << std::endl;
    } catch (const char *ex) {
        std::cout << ex << std::endl;
    } catch (...) {
        std::cout << "Unknown exception" << std::endl;
    }

    return 0;
}
