// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef WINOGRADECONV_UTLS_H
#define WINOGRADECONV_UTLS_H

float* GetInput();

float* GetWeight();

float* GetBias();

void WriteOutput(float* output, int count);

enum CVT_DIR { NHWC2NCHW, NCHW2NHWC };
template <class T>
static bool ConvertBetweenNHWCAndNCHW(T* src, T* dst, int num, int channel, int height, int width, CVT_DIR dir) {
  assert(dir == NHWC2NCHW || dir == NCHW2NHWC);
  bool alloc_mem = false;
  if (dst == nullptr) {
    alloc_mem = true;
    dst = new T[num * channel * height * width]();
  }

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          // n * channel * height * width + c * height * width + h * width + w
          // n * height * width * channel + h * width * channel + w * channel + c
          if (NHWC2NCHW == dir) {
            // nhwc -> nchw
            dst[n * channel * height * width + c * height * width + h * width + w] =
                src[n * height * width * channel + h * width * channel + w * channel + c];
          } else {
            // nchw -> nhwc
            dst[n * height * width * channel + h * width * channel + w * channel + c] =
                src[n * channel * height * width + c * height * width + h * width + w];
          }
        }
      }
    }
  }
  if (alloc_mem) {
    memcpy(src, dst, num * channel * height * width * sizeof(T));
    delete[] dst;
  }
  return true;
}

#endif  // WINOGRADECONV_UTLS_H
