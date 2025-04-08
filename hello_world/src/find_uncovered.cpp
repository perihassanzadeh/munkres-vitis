#include "hls_stream.h"
#include "ap_int.h"
#include <stdbool.h>

#define MAX_SIZE 256

extern "C" {

void find_uncovered_kernel(
    const double matrix[MAX_SIZE][MAX_SIZE],
    const int mask_matrix[MAX_SIZE][MAX_SIZE],
    const bool row_mask[MAX_SIZE],
    const bool col_mask[MAX_SIZE],
    const double item,
    ap_uint<32> size,
    ap_uint<32>* row_out,
    ap_uint<32>* col_out,
    bool* found
) {
#pragma HLS INTERFACE m_axi port = matrix offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = mask_matrix offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = row_mask offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = col_mask offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = item bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE m_axi port = row_out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = col_out offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = found offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = return bundle = control

    *found = false;
    
    row_loop: for (ap_uint<32> row = 0; row < size; row++) {
        if (!row_mask[row]) {
            col_loop: for (ap_uint<32> col = 0; col < size; col++) {
                if (!col_mask[col]) {
                    if (matrix[row][col] == item) {
                        *row_out = row;
                        *col_out = col;
                        *found = true;
                        return;
                    }
                }
            }
        }
    }
}

} 
