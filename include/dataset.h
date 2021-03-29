#ifndef DATASET_H
#define DATASET_H

#include "Eigen/Core"
#include "Eigen/Dense"
#include "eigen_matrix_io.h"
#include "general.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace KNet {
// каждый тип должен предоставить свою реализацию версий
// void Write(std::ostream&, const T&)
// void Read(std::istream&, T&);
// T - применяемый тип

template<typename DType>
class DataSet {
public:
    DataSet();
    DataSet(const std::vector<DType> &);
    DataSet(const DataSet &);
    ~DataSet();

    DataSet &operator=(const DataSet &);

    void Push(const DType &);
    void Push(DType &&);
    void Pop();

    std::size_t Size() const;

    const DType &operator[](Eigen::Index) const;
    DType &operator[](Eigen::Index);

    void MixData();

    void Write(const std::string &) const;
    void Read(const std::string &);

private:
    std::vector<DType> data;
};

// Specializations

typedef Eigen::MatrixXd KNetMatrixXd;

// Declarations

template<typename DType>
DataSet<DType>::DataSet() {
}

template<typename DType>
DataSet<DType>::DataSet(const std::vector<DType> &UD)
    : data(UD) {
}

template<typename DType>
DataSet<DType>::DataSet(const DataSet<DType> &UD)
    : data(UD.data) {
}

template<typename DType>
DataSet<DType>::~DataSet() {
}

template<typename DType>
DataSet<DType> &DataSet<DType>::operator=(const DataSet<DType> &UD) {
    if (this == &UD) {
        return *this;
    }

    data = UD.data;

    return *this;
}

template<typename DType>
void DataSet<DType>::Push(const DType &UD) {
    data.push_back(UD);
}

template<typename DType>
void DataSet<DType>::Push(DType &&UD) {
    data.push_back(std::move(UD));
}

template<typename DType>
void DataSet<DType>::Pop() {
    if (data.empty()) {
        throw KNet::generate_exception("Couldn't pop data - the dataset is empty", __LINE__, __FILE__);
    }
    data.pop_back();
}

template<typename DType>
std::size_t DataSet<DType>::Size() const {
    return data.size();
}

template<typename DType>
void DataSet<DType>::MixData() {
    std::vector<int> Indexes;
    std::vector<int> Temp(data.size());

    for (int i = 0; i < data.size(); i++) {
        Temp[i] = i;
    }

    while (Indexes.size() < data.size()) {
        int ind = rand() % Temp.size();
        Indexes.push_back(Temp[ind]);
        Temp.erase(Temp.begin() + ind);
    }

    std::vector<DType> Result(data.size());

    for (std::size_t i = 0; i < Indexes.size(); i++) {
        Result[i] = data[Indexes[i]];
    }

    data = Result;
}

template<typename DType>
const DType &DataSet<DType>::operator[](Eigen::Index i) const {
    if (i < 0 || i >= data.size()) {
        throw KNet::generate_exception("Out of range in dataset", __LINE__, __FILE__);
    }

    const DType *casted = dynamic_cast<const DType *>(&data[i]);
    if (casted == nullptr) {
        throw KNet::generate_exception("Couldn't cast to chosen type", __LINE__, __FILE__);
    }

    return *casted;
}

template<typename DType>
DType &DataSet<DType>::operator[](Eigen::Index i) {
    if (i < 0 || i >= data.size()) {
        throw KNet::generate_exception("Out of range in dataset", __LINE__, __FILE__);
    }

    DType *casted = dynamic_cast<DType *>(&data[i]);
    if (casted == nullptr) {
        throw KNet::generate_exception("Couldn't cast to chosen type", __LINE__, __FILE__);
    }

    return *casted;
}

template<typename DType>
void DataSet<DType>::Write(const std::string &filename) const {
    std::ofstream ostr(filename, std::ios_base::binary | std::ios_base::out);

    if (!ostr.is_open()) {
        throw KNet::generate_exception(std::string("Couldn't open file \'") + filename + "\'", __LINE__, __FILE__);
    }

    std::size_t sz = data.size();

    ostr.write(reinterpret_cast<char *>(&sz), sizeof(sz));
    for (std::size_t i = 0; i < sz; i++) {
        Write(ostr, data[i]);
    }
}

template<typename DType>
void DataSet<DType>::Read(const std::string &filename) {
    std::ifstream istr(filename, std::ios_base::binary | std::ios_base::in);

    if (!istr.is_open()) {
        throw KNet::generate_exception(std::string("Couldn't open file \'") + filename + "\'", __LINE__, __FILE__);
    }

    std::size_t sz;

    istr.read(reinterpret_cast<char *>(&sz), sizeof(sz));
    data.clear();
    data.resize(sz);

    for (std::size_t i = 0; i < sz; i++) {
        Read(istr, data[i]);
    }
}
}// namespace KNet

#endif
