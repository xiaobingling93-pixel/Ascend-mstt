#pragma once

#include <string>

#define CONFIG_EXAMPLE __RESOURCES_PATH__"/config.json"

std::string TEST_ExecShellCommand(const std::string& cmd);
std::string Trim(const std::string& str);
