/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime_api.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(val) \
  { nv::cuda_check_((val), __FILE__, __LINE__); }

namespace nv {

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<__hip_fp8_e4m3> : std::true_type {};

template <>
struct is_fp8<__hip_fp8_e5m2> : std::true_type {};

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(hipError_t val, const char* file, int line) {
  if (val != hipSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) + ": CUDA error " +
                        std::to_string(val) + ": " + hipGetErrorString(val));
  }
}

class CudaDeviceRestorer {
 public:
  CudaDeviceRestorer() { CUDA_CHECK(hipGetDevice(&dev_)); }
  ~CudaDeviceRestorer() { CUDA_CHECK(hipSetDevice(dev_)); }
  void check_device(int device) const {
    if (device != dev_) {
      throw std::runtime_error(
          std::string(__FILE__) + ":" + std::to_string(__LINE__) +
          ": Runtime Error: The device id in the context is not consistent with configuration");
    }
  }

 private:
  int dev_;
};

inline int get_dev(const void* ptr) {
  hipPointerAttribute_t attr;
  CUDA_CHECK(hipPointerGetAttributes(&attr, ptr));
  int dev = -1;

#if CUDART_VERSION >= 10000
  if (attr.type == hipMemoryTypeDevice)
#else
  if (attr.memoryType == hipMemoryTypeDevice)
#endif
  {
    dev = attr.device;
  }
  return dev;
}

inline void switch_to_dev(const void* ptr) {
  int dev = get_dev(ptr);
  if (dev >= 0) {
    CUDA_CHECK(hipSetDevice(dev));
  }
}

}  // namespace nv
