#ifndef GENERAL_H
#define GENERAL_H

#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

namespace KNet {
std::runtime_error generate_exception(const std::string &mess, int Line = -1, const char *filename = nullptr);
}

#endif
