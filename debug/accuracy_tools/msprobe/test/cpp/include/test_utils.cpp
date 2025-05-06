#include <iostream>
#include <array>
#include <memory>
#include <cstdio>

std::string TEST_ExecShellCommand(const std::string& cmd)
{
    std::array<char, 1024> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string Trim(const std::string& str)
{
    std::string::size_type first = str.find_first_not_of(" \t\n\r\f\v");
    std::string::size_type last = str.find_last_not_of(" \t\n\r\f\v");
    if (first == std::string::npos || last == std::string::npos) {
        return "";
    }
    
    return str.substr(first, (last - first + 1));
}
