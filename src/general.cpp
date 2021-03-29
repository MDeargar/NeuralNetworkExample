#include "../include/general.h"

namespace KNet {
std::runtime_error generate_exception(const std::string &mess, int Line, const char *filename) {
    std::string add;
    if (Line != -1) {
        std::ostringstream buff;
        buff << Line;
        add = std::string("\n\t") + buff.str() + ": ";
    }

    return std::runtime_error(((filename == nullptr) ? "" : std::string(filename)) + add + mess);
}
}// namespace KNet
