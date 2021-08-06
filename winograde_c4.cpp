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

#include "winograde_c4.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>

/**
 *  U   = GgG^T = (4, 4)
 *  G   = (4,3)
 *  g   = (3,3)
 *  G^T = (3,4)
 *
 * weight: {OC, IC, 3, 3} --> {R(OC, 4), R(IC, 16), 4, 4}
 *
 * weight:
 *      size:   {R(OC, 4),  R(IC, 16), 4, 4}
 *      format: {R(OC,4)/4, 4,  4,  R(IC, 16),  4,  16}
 *                          |   |               |   |
 *                         KH  KW               OC  IC
 * */
void weight_convert(float* dst, const float* src) {
  float G[4][3]  = {{1, 0, 0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0, 0, 1}};
  float GT[3][4] = {{1, 0.5, 0.5, 0}, {0, 0.5, -0.5, 0}, {0, 0.5, 0.5, 1}};

  int IC            = 16;
  int OC            = 16;
  int KH            = 3;
  int KW            = 3;
  int IC_R16        = ROUND_UP(IC, 16);
  int OC_R4         = ROUND_UP(OC, 4);
  float w[3][3]     = {0.0f};
  float mid[4][3]   = {0.0f};
  float win_w[4][4] = {0.0f};
  for (int oc = 0; oc < OC; ++oc) {
    // loop for IC_R16, meaning, pad IC to 16
    for (int ic = 0; ic < IC_R16; ++ic) {
      if (ic < IC) {
        // get weight kernel: {3, 3}
        for (int i = 0; i < KH; ++i) {
          for (int j = 0; j < KW; ++j) {
            int index = oc * IC * KH * KW + ic * KH * KW + i * KW + j;
            w[i][j]   = src[index];
          }
        }
        // Gxg
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 3; ++j) {
            mid[i][j] = G[i][0] * w[0][j] + G[i][1] * w[1][j] + G[i][2] * w[2][j];
          }
        }
        // GxgxG^T
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            win_w[i][j] = mid[i][0] * GT[0][j] + mid[i][1] * GT[1][j] + mid[i][2] * GT[2][j];
          }
        }
        // reformat weight
        int ic_m16 = ic % 16;
        int ic_d4  = ic / 16;
        int oc_m4  = oc % 4;
        int oc_d4  = oc / 4;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            int index  = oc_d4 * 4 * IC_R16 * 4 * 4 + (i * 4 + j) * IC_R16 * 4 + ic_d4 * 4 * 16 + oc_m4 * 16 + ic_m16;
            dst[index] = win_w[i][j];
          }
        }
      } else {
        // process ic >= IC, pad channel
        int ic_m16 = ic % 16;
        int ic_d4  = ic / 16;
        int oc_m4  = oc % 4;
        int oc_d4  = ic / 4;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            int index  = oc_d4 * 4 * IC_R16 * 4 * 4 + (i * 4 + j) * IC_R16 * 4 + ic_d4 * 4 * 16 + oc_m4 * 16 + ic_m16;
            dst[index] = 0;
          }
        }
      }
    }
  }
}

void input_convert(float* wino_input_tile, const float* src_tile, const int width, const int IC) {
  int ic_r16                 = ROUND_UP(IC, 16);
  int w_step                 = 4 * ic_r16;
  int h_step                 = 4 * w_step;
  float v[4][4][16]          = {0.0f};
  float mid[4][4][16]        = {0.0f};
  float wino_input[4][4][16] = {0.0f};
  for (int ic = 0; ic < IC; ic += 16) {
    for (int h = 0; h < 4; ++h) {
      for (int w = 0; w < 4; ++w) {
        for (int c = 0; c < 16; ++c) {
          v[h][w][c] = src_tile[h * width * IC + w * IC + c];
        }
      }
    }
    /**
     * B^Txd
     * */
    int BT[4][4] = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
    int B[4][4]  = {{1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
    for (int h = 0; h < 4; ++h) {
      for (int w = 0; w < 4; ++w) {
        for (int c = 0; c < 16; ++c) {
          for (int k = 0; k < 4; ++k) {
            mid[h][w][c] += BT[h][k] * v[k][w][c];
          }
        }
      }
    }
    /**
     * B^TxdxB
     * */
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int c = 0; c < 16; ++c) {
          for (int k = 0; k < 4; ++k) {
            wino_input[i][j][c] += mid[i][k][c] * B[k][j];
          }
        }
      }
    }
    // [4, 4] (2, 2) * IC_R16
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int c = 0; c < 16; ++c) {
          /**
           * step1 : (2x2) * ic_r16
           * step2 : 4 * (2x2) * ic_r16
           * */
          wino_input_tile[i * h_step + j * w_step + c] = wino_input[i][j][c];
        }
      }
    }
  }
}

/**
 * Hadamard product
 * */
void HadamardProduct(float* hadamard, const float* wino_input, const float* wino_weight, const int IC) {
  /**
   *  temp: (2, 2, 4)
   *         |  |  |
   *         oh ow oc
   **/
  float temp[2][2][4] = {0.0f};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int oc = 0; oc < 4; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
          auto w = wino_weight[oc * IC + ic];
          auto v = wino_input[i * 2 * IC + j * IC + ic];
          temp[i][j][oc] +=  w * v;
        }
      }
    }
  }
  // reformat
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 4; ++k) {
        hadamard[i * 2 * 4 + j * 4 + k] = temp[i][j][k];
      }
    }
  }
}

/**
 * A^T{(GgG^T) ()}A
 *
 * */

void dst_convert(float* output, const float* src, int h_stride, int w_stride, int h_cnt, int w_cnt) {
  int AT[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};
  int A[4][2]  = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
  // 4x4xOC
  float v[4][4][4]     = {0.0f};
  float mid_w[2][4][4] = {0.0f};
  float dst_w[2][2][4] = {0.0f};
  for (int h = 0; h < 4; ++h) {
    for (int w = 0; w < 4; ++w) {
      for (int c = 0; c < 4; ++c) {
        /**
         * wino format: 对吗?
         * wino_src format: (4,4)x(2,2)x4
         *                              |
         *                              oc
         * */
        v[h][w][c] = src[(h * 16 + w * 4) * 4 + c];
      }
    }
  }
  /**
   * A^TM:  (2,4)x(4,4)x(4) --> (2, 4)x(4)
   * */
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int oc = 0; oc < 4; ++oc) {
          mid_w[i][j][oc] += AT[i][k] * v[k][j][oc];
        }
      }
    }
  }
  /**
   * A^TMA: (2, 4)(4, 2) (4) --> (2,2)x(4)
   * */
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int oc = 0; oc < 4; ++oc) {
          dst_w[i][j][oc] += mid_w[i][k][oc] * A[k][j];
        }
      }
    }
  }
  /**
   * w_stride: OC
   * h_stride: OW*OC
   *
   * */
  for (int i = 0; i < 4; ++i) {
    output[i] = dst_w[0][0][i];
  }
  if (w_cnt > 1) {
    for (int i = 0; i < 4; ++i) {
      output[w_stride + i] = dst_w[0][1][i];
    }
  }
  if (h_cnt > 1) {
    for (int i = 0; i < 4; ++i) {
      output[h_stride + i] = dst_w[1][0][i];
    }
  }
  if (w_cnt > 1 && h_cnt > 1) {
    for (int i = 0; i < 4; ++i) {
      output[h_stride + w_stride + i] = dst_w[1][1][i];
    }
  }
}

/**
 * winograde
 * Y = A^T[ (GgG^T) hadamard (B^TdB)]A
 *
 * */
void WinogradeNHWC(float* output, const float* input, const float* weight, const float* bias) {
  int pad          = 1;
  int N            = 1;
  int IC           = 16;
  int IHP          = 6;
  int IWP          = 6;
  int OC           = 16;
  int OH           = 4;
  int OW           = 4;
  int IC_R16       = ROUND_UP(IC, 16);
  int OC_R4        = ROUND_UP(OC, 4);
  int hw_tile_cnt  = 2 * 2;
  int hw_tile_size = 4 * 4;
  int weight_tile  = 4 * 4;
  /**
   * weight: {OC, IC, 3, 3} --> {R(OC, 4), R(IC, 16), 4, 4}
   * weight:
   *      size:   {R(OC, 4),  R(IC, 16), 4, 4}
   *      format: {R(OC,4)/4, [4, 4],  R(IC, 16),  4,  16}
   *                          |   |                |   |
   *                         KH  KW               OC  IC
   * */
  auto wino_weight_buffer = (float*)calloc(OC_R4 * weight_tile * IC_R16, sizeof(float));
  /**
   * input tile buffer:
   *        size:   (2x2)x(4x4)xIC_R16
   *        format: {[4,4], R(IC, 16)/16, (2, 2), 16}
   *                                       |  |   |
   *                                       OH,OW  IC
   * */
  auto wino_input_buffer = (float*)calloc(hw_tile_cnt * hw_tile_size * IC_R16, sizeof(float));
  /**
   * hadamard_buffer:
   *        size:   (4x4)x(2,2)x(4)
   *        format: {[4,4], (2x2), (4)}
   *                          |     |
   *                        OH,OW   OC
   * */
  auto hadamard_buffer = (float*)calloc(hw_tile_size * hw_tile_size * 4, sizeof(float));

  weight_convert(wino_weight_buffer, weight);
  for (int oh = 0; oh < OH; oh += 4) {    // process 2 tile for OH
    for (int ow = 0; ow < OW; ow += 4) {  // process 2 tile for O
      for (int ht = 0; ht < 2; ++ht) {
        for (int wt = 0; wt < 2; ++wt) {
          int ih_stride         = (oh + ht * 2) * IWP * IC;
          int iw_stride         = (ow + wt * 2) * IC;
          auto input_tile_base  = input + ih_stride + iw_stride;
          auto wino_tile_stride = (ht * 2 + wt) * hw_tile_size;
          auto wino_tile_base   = wino_input_buffer + wino_tile_stride;
          input_convert(wino_tile_base, input_tile_base, IWP, IC);
        }
      }
      for (int oc = 0; oc < OC; oc += 4) {
        /**
         * Hadamard product
         * */
        for (int win_tile = 0; win_tile < 16; ++win_tile) {
          auto wino_input_tile = wino_input_buffer + win_tile * (2 * 2) * IC_R16;
          // error: missing wintile
          auto wino_weight_tile =wino_weight_buffer + win_tile * 4 * IC_R16 + oc * 16* IC_R16;
          auto hadamard_buffer_tile = hadamard_buffer + win_tile * 16;
          HadamardProduct(hadamard_buffer_tile, wino_input_tile, wino_weight_tile, IC_R16);
        }
        for (int ht = 0; ht < 2; ++ht) {
          for (int wt = 0; wt < 2; ++wt) {
            if (ow + wt * 2 >= OW || oh + ht * 2 >= OH) {
              continue;
            }
            // 代表最后的数据排布
            float* output_tile_base     = output + (oh + ht * 2) * OW * OC + (ow + wt * 2) * OC + oc;
            float* hadamard_buffer_tile = hadamard_buffer + (ht * 2 + wt) * 4;
            int h_stride                = OW * OC;
            int w_stride                = OC;
            int h_cnt                   = OH - (oh + ht * 2);
            int w_cnt                   = OW - (ow + wt * 2);
            dst_convert(output_tile_base, hadamard_buffer_tile, h_stride, w_stride, h_cnt, w_cnt);
          }
        }
      }
    }
  }
  // process bias
  for (int oc = 0; oc < OC; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        output[oh * OW * OC + ow * OC + oc] += bias[oc];
      }
    }
  }
  free(wino_weight_buffer);
  free(wino_input_buffer);
  free(hadamard_buffer);
}
