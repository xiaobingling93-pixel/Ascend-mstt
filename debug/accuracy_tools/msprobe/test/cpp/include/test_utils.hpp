#pragma once

#include <string>

#define CONFIG_EXAMPLE __RESOURCES_PATH__"/config.json"

std::string TEST_ExecShellCommand(const std::string& cmd);
std::string trim(const std::string& str);
