// SPDX-License-Identifier: MIT
/* Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. */

#pragma once

#include <cstdio>

#include <fmt/core.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern "C" {
extern char** environ;
}

namespace maestro {

namespace detail {

inline void print_env_variables() {
  char** env = environ;
  while (*env) {
    std::cout << *env << std::endl;
    ++env;
  }
}

template <typename... Args>
inline void log_message(const char* level,
                        const char* file,
                        int line,
                        const char* msg,
                        Args... args) {
  static const char* log_env = std::getenv("TRACER_LOG_LEVEL");
  if (log_env) {
    static std::string log_env_str = std::string(log_env);
    std::transform(log_env_str.begin(), log_env_str.end(), log_env_str.begin(),
                   [](unsigned char c) { return std::toupper(c); });

    if (std::string(log_env_str) == level) {
      const char* color_reset = "\033[0m";
      const char* color_info = "\033[37m";
      const char* color_error = "\033[31m";

      const char* color =
          (std::strcmp(level, "ERROR") == 0) ? color_error : color_info;

      std::string formatted_message;
      if constexpr (sizeof...(args) > 0) {
        formatted_message = fmt::vformat(msg, fmt::make_format_args(args...));
      } else {
        formatted_message = msg;
      }

      std::printf("%s%s: [%s:%d] %s%s\n", color, level, file, line,
                  formatted_message.c_str(), color_reset);

      static const char* log_file = std::getenv("TRACER_LOG_FILE");
      if (log_file) {
        static std::ofstream log_stream(log_file, std::ios::app);
        if (log_stream) {
          std::ostringstream oss;
          oss << level << ": [" << file << ":" << line << "] "
              << formatted_message << "\n";
          log_stream << oss.str();
        }
      }
    }
  }
}
}  // namespace detail
}  // namespace maestro

#define LOG_DETAIL(msg, ...) \
  maestro::detail::log_message("DETAIL", __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) \
  maestro::detail::log_message("INFO", __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) \
  maestro::detail::log_message("ERROR", __FILE__, __LINE__, msg, ##__VA_ARGS__)
