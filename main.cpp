#include <iostream>

#include "utls.h"

/**
 * F(2x2, 3x3)
 *  U   = GgG^T
 *  G   = (4,3)
 *  g   = (3,3)
 *  G^T = (3,4)
 *  U = GxgxG^T
 * */

void GgGT(float* winograde_weight, float* kernel) {
  float G[4][3] = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
  float GT[3][4] = {{1, 0.5, 0.5, 0}, {0, 0.5, -0.5, 0}, {0, 0.5, 0.5, 1}};
  float Gxg[4][3] = {0.0f};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      float temp = 0.0f;
      for (int k = 0; k < 3; ++k) {
        temp += G[i][k] * kernel[k * 3 + j];
      }
      Gxg[i][j] = temp;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float temp = 0.0f;
      for (int k = 0; k < 3; ++k) {
        temp += Gxg[i][k] * GT[k][j];
      }
      winograde_weight[i * 4 + j] = temp;
    }
  }
}
/**
 *  V   = B^TxdxB
 *  B^T = (4, 4)
 *  d   = (4, 4)
 *  B   = (4, 4)
 * */

void BTdB(float* wino_input, float* input, int start, int IW) {
  int BT[4][4] = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
  int B[4][4] = {{1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
  float BT_B[4][4] = {0.0f};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float temp = 0.0f;
      for (int k = 0; k < 4; ++k) {
        temp += BT[i][k] * input[k * IW + j + start];  // keypoint: k * IW
      }
      BT_B[i][j] = temp;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      float temp = 0.0f;
      for (int k = 0; k < 4; ++k) {
        temp += BT_B[i][k] * B[k][j];
      }
      wino_input[i * 4 + j] = temp;
    }
  }
}

/**
 * M = A^Tx[U hadamard V]xA
 * A^T  = (2, 4)
 * U    = (4, 4)
 * V    = (4, 4)
 * A    = (4, 2)
 * Y    = (2, 2)
 * */
void output_convert(float* Y, float* U, float* V) {
  int AT[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
  int A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};

  float M[4][4] = {0.0f};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      M[i][j] = U[i * 4 + j] * V[i * 4 + j];
    }
  }
  float AT_M[2][4] = {0.0f};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      float temp = 0.0f;
      for (int k = 0; k < 4; ++k) {
        temp += AT[i][k] * M[k][j];
      }
      AT_M[i][j] = temp;
    }
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      float temp = 0.0f;
      for (int k = 0; k < 4; ++k) {
        temp += AT_M[i][k] * A[k][j];
      }
      Y[i * 2 + j] = temp;
    }
  }
}

/**
 * winograde
 * Y = A^T[ (GgG^T) hadamard (B^TdB)]A
 *
 * */

void winograde(float* output, float* input, float* weight, float* bias) {
  int N = 1;
  int IC = 16;
  int IH = 6;
  int IW = 6;
  int OC = 16;
  int OH = 4;
  int OW = 4;
  int pad = 1;
  int kernel_size = 3;
  auto* wino_weight = (float*)calloc(4 * 4, sizeof(float));
  auto* wino_input = (float*)calloc(4 * 4, sizeof(float));
  auto* wino_output = (float*)calloc(2 * 2, sizeof(float));
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; oh += 2) {
      for (int ow = 0; ow < OW; ow += 2) {
        float temp[4];
        std::fill_n(temp, 4, bias[oc]);
        for (int ic = 0; ic < IC; ++ic) {
          float* kernel = weight + oc * IC * kernel_size * kernel_size + ic * kernel_size * kernel_size;
          // wino_weight: (4,4)
          GgGT(wino_weight, kernel);
          int input_start = ic * IH * IW + oh * IW + ow;
          // wino_input: (4,4)
          BTdB(wino_input, input, input_start, IW);
          // wino_output: 2x2
          output_convert(wino_output, wino_weight, wino_input);

          for (int i = 0; i < 4; ++i) {
            temp[i] += wino_output[i];
          }
        }
        int output_index = oc * OH * OW + oh * OW + ow;
        std::cout << "output_index: " << output_index << std::endl;
        output[output_index] = temp[0];
        output[output_index + 1] = temp[1];
        output[output_index + OW] = temp[2];
        output[output_index + OW + 1] = temp[3];
      };
    }
  }
  free(wino_input);
  free(wino_weight);
  free(wino_output);
}

/**
 * pad = 1
 *
 * */
void padding(float* input, float* paded_input, int IC, int IH, int IW, int pad) {
  int PIH = IH + 2 * pad;
  int PIW = IW + 2 * pad;
  int t = pad;
  int l = pad;
  int r = IW + pad;
  int b = IH + pad;
  for (int ic = 0; ic < IC; ++ic) {
    for (int ih = 0; ih < PIH; ++ih) {
      for (int iw = 0; iw < PIW; ++iw) {
        if (ih < t || ih >= b || iw < l || iw >= r) {
          paded_input[ic * PIH * PIW + ih * PIW + iw] = 0;
        } else {
          int input_index = ic * IH * IW + (ih - 1) * IW + iw - 1;
          int pad_input_index = ic * PIH * PIW + ih * PIW + iw;
          paded_input[pad_input_index] = input[input_index];
        }
      }
    }
  }
}

int main() {
  /**
   * input:     {1, 16, 4, 4}
   * weight:    {16, 16, 3, 3}
   * bias:      {16}
   * output:    {1, 16, 4, 4}
   * pad:       {1, 1, 1, 1}
   * stride:    {1, 1}
   * group :    {1}
   * */
  const int input_size = 4;
  const int kernel_size = 3;
  int N = 1;
  int IC = 16;
  int IH = 4;
  int IW = 4;
  int OC = 16;
  int OH = 4;
  int OW = 4;
  int pad = 1;

  // input: {n, IC, IH, IW} = {1, 16, 4, 4}
  float* input = GetInput();
  auto* pad_input = (float*)calloc(N * IC * (IH + 2 * pad) * (IW + 2 * pad), sizeof(float));
  // pad_input: {n, IC, IH, IW} = {1, 16, 6, 6}
  padding(input, pad_input, IC, IH, IW, pad);
  //ConvertBetweenNHWCAndNCHW<float>(pad_input, nullptr, N, IC, (IH + 2), (IW + 2), NCHW2NHWC);
  free(input);
  // weight: {OC, IC, KH, KW} = {16, 16, 3, 3}
  float* weight = GetWeight();
  // bias: {OC} = {16}
  float* bias = GetBias();
  auto output = (float*)calloc(N * OC * OH * OW, sizeof(float));
  winograde(output, pad_input, weight, bias);
  // nhwc -> nchw
  //ConvertBetweenNHWCAndNCHW<float>(output, nullptr, N, OC, OH, OW, NHWC2NCHW);
  WriteOutput(output, N * OC * OH * OW);
  free(pad_input);
  free(output);
  return 0;
}
