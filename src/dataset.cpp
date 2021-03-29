#include "../include/dataset.h"

namespace KNet {
void Write(std::ostream &out, const Eigen::MatrixXd &M) {
    Eigen::write_binary(out, M);
}

void Read(std::istream &in, Eigen::MatrixXd &M) {
    Eigen::read_binary(in, M);
}

void Write(std::ostream &ostr, const std::pair<Eigen::MatrixXd, Eigen::MatrixXd> &P) {
    Eigen::write_binary(ostr, P.first);
    Eigen::write_binary(ostr, P.second);
}

void Read(std::istream &istr, std::pair<Eigen::MatrixXd, Eigen::MatrixXd> &P) {
    Eigen::read_binary(istr, P.first);
    Eigen::read_binary(istr, P.second);
}

void Write(std::ostream &out, const std::tuple<int, int, int> &T) {
    out.write(reinterpret_cast<const char *>(&std::get<0>(T)), sizeof(int));
    out.write(reinterpret_cast<const char *>(&std::get<1>(T)), sizeof(int));
    out.write(reinterpret_cast<const char *>(&std::get<2>(T)), sizeof(int));
}

void Read(std::istream &istr, std::tuple<int, int, int> &T) {
    istr.read(reinterpret_cast<char *>(&std::get<0>(T)), sizeof(int));
    istr.read(reinterpret_cast<char *>(&std::get<1>(T)), sizeof(int));
    istr.read(reinterpret_cast<char *>(&std::get<2>(T)), sizeof(int));
}

void Write(std::ostream &out, const std::tuple<double, std::size_t> &T) {
    out.write(reinterpret_cast<const char *>(&std::get<0>(T)), sizeof(double));
    out.write(reinterpret_cast<const char *>(&std::get<1>(T)), sizeof(std::size_t));
}

void Read(std::istream &istr, std::tuple<double, std::size_t> &T) {
    istr.read(reinterpret_cast<char *>(&std::get<0>(T)), sizeof(double));
    istr.read(reinterpret_cast<char *>(&std::get<1>(T)), sizeof(std::size_t));
}
}// namespace KNet
