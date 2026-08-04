// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

class Stats
{
public:
    uint64_t transfer_time = 0;
    uint64_t compute_time = 0;
    uint64_t comm_time = 0;

    uint64_t conv_time = 0;
    uint64_t conv_compute_time = 0;
    uint64_t conv_comm_time = 0;

    uint64_t matmul_time = 0;
    uint64_t matmul_compute_time = 0;
    uint64_t matmul_comm_time = 0;

    uint64_t relu_time = 0;
    uint64_t reluext_time = 0;
    uint64_t reluext_comm_time = 0;

    uint64_t maxpool_time = 0;
    uint64_t maxpool_comm_time = 0;
    uint64_t avgpool_time = 0;

    uint64_t truncate_time = 0;
    uint64_t truncate_comm_time = 0;
    uint64_t signext_time = 0;

    uint64_t gelu_time = 0;
    uint64_t layernorm_time = 0;
    uint64_t softmax_time = 0;
    uint64_t mha_time = 0;

    uint64_t linear_comm_bytes = 0;
    uint64_t gelu_comm_bytes = 0;
    uint64_t softmax_comm_bytes = 0;
    uint64_t layernorm_comm_bytes = 0;

    // extra stats for detailed profiling
    
    uint64_t mha_matmul_transfer_time = 0; // OpType flag 1
    uint64_t mha_matmul_compute_time = 0;
    uint64_t mha_matmul_comm_time = 0;

    uint64_t mha_softmax_transfer_time = 0; // OpType flag 2
    uint64_t mha_softmax_compute_time = 0;
    uint64_t mha_softmax_comm_time = 0;

    uint64_t mha_rot_transfer_time = 0; // OpType flag 3
    uint64_t mha_rot_compute_time = 0;
    uint64_t mha_rot_comm_time = 0;

    uint64_t layernorm_transfer_time = 0; // OpType flag 4
    uint64_t layernorm_compute_time = 0;
    uint64_t layernorm_comm_time = 0;

    uint64_t dcf_transfer_time = 0; // OpType flag 5
    uint64_t dcf_compute_time = 0; 
    uint64_t dcf_comm_time = 0; 

    uint64_t truncate_global_transfer_time = 0; // Self measured in the function definition, not using the OpType flag
    uint64_t truncate_global_compute_time = 0;
    uint64_t truncate_global_comm_time = 0;

    uint64_t truncate_matmul_transfer_time = 0; // the same method as above // this is only for MHA
    uint64_t truncate_matmul_compute_time = 0;
    uint64_t truncate_matmul_comm_time = 0;

    uint64_t QKV_transfer_time = 0;
    uint64_t QKV_compute_time = 0;
    uint64_t QKV_comm_time = 0;

    uint64_t mha_proj_transfer_time = 0;
    uint64_t mha_proj_compute_time = 0;
    uint64_t mha_proj_comm_time = 0;    

    void reset()
    {
        transfer_time = 0;
        compute_time = 0;
        comm_time = 0;

        conv_time = 0;
        conv_compute_time = 0;
        conv_comm_time = 0;

        matmul_time = 0;
        matmul_compute_time = 0;
        matmul_comm_time = 0;

        relu_time = 0;
        reluext_time = 0;
        reluext_comm_time = 0;

        maxpool_time = 0;
        maxpool_comm_time = 0;

        avgpool_time = 0;
        
        truncate_time = 0;
        truncate_comm_time = 0;

        signext_time = 0;

        gelu_time = 0;
        layernorm_time = 0;
        softmax_time = 0;
        layernorm_comm_bytes = 0;
        linear_comm_bytes = 0;
        softmax_comm_bytes = 0;
        gelu_comm_bytes = 0;
        mha_time = 0;
        
        // the extra stats

        mha_matmul_transfer_time = 0;
        mha_matmul_compute_time = 0;
        mha_matmul_comm_time = 0;

        mha_softmax_transfer_time = 0;
        mha_softmax_compute_time = 0;
        mha_softmax_comm_time = 0;

        mha_rot_transfer_time = 0;
        mha_rot_compute_time = 0;
        mha_rot_comm_time = 0;

        layernorm_transfer_time = 0;
        layernorm_compute_time = 0;
        layernorm_comm_time = 0;

        dcf_transfer_time = 0;
        dcf_compute_time = 0;
        dcf_comm_time = 0;

        truncate_global_transfer_time = 0;
        truncate_global_compute_time = 0;
        truncate_global_comm_time = 0;

        truncate_matmul_transfer_time = 0;
        truncate_matmul_compute_time = 0;
        truncate_matmul_comm_time = 0;

        QKV_transfer_time = 0;
        QKV_compute_time = 0;
        QKV_comm_time = 0;

        mha_proj_transfer_time = 0;
        mha_proj_compute_time = 0;
        mha_proj_comm_time = 0;
    }
};