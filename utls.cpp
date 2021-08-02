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
#include <fstream>
#include <iostream>
#include <istream>
#include <vector>
float* GetInput() {
  std::string file_path = "/Users/tiankai/git-hub/TNN/tools/convert2tnn/temp_data/input.txt";

  std::ifstream f_stream(file_path.c_str());
  int count = 1;
  int dims_size = 0;
  std::string input_name;
  std::vector<int> dims;
  int data_type = 0;
  f_stream >> count;
  f_stream >> input_name;
  f_stream >> dims_size;
  int input_size = 1;
  for (int i = 0; i < dims_size; ++i) {
    uint32_t dim;
    f_stream >> dim;
    dims.push_back(dim);
    input_size *= dim;
  }
  f_stream >> data_type;
  auto input_data = (float *)calloc(input_size, sizeof(float ));
  float temp;
  for (int i = 0; i < input_size; ++i) {
    f_stream >> temp;
    input_data[i] = temp;
  }
  f_stream.close();
  return input_data;
}

float* GetWeight() {
  std::string weight_path = "/Users/tiankai/tmp/conv_weight.txt";
  std::ifstream  f_stream(weight_path);
  int weight_size = 16*16*3*3;
  auto weight = (float *)calloc(weight_size, sizeof(float ));
  float temp = 0.0f;
  for (int i = 0; i < weight_size; ++i) {
    f_stream >> temp;
    weight[i] = temp;
  }
  f_stream.close();
  return weight;
}

float* GetBias() {
  std::string bias_path = "/Users/tiankai/tmp/conv_bias.txt";
  std::ifstream  f_stream(bias_path);
  int bias_size = 16;
  auto bias = (float *)calloc(bias_size, sizeof(float ));
  float temp = 0.0f;
  for (int i = 0; i < bias_size; ++i) {
    f_stream >> temp;
    bias[i] = temp;
  }
  f_stream.close();
  return bias;
}

void WriteOutput(float* output, int count) {
  std::string output_path = "/Users/tiankai/git-hub/TNN/tools/convert2tnn/temp_data/wino_output.txt";
  std::ofstream f_stream(output_path);
  for (int i = 0; i < count; ++i) {
    f_stream << output[i] << std::endl;
  }
  f_stream.close();
}