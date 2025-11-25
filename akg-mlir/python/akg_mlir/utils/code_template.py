cuda_runtime_template = '''
/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cuda.h>
#include <cuda_runtime.h>
#include<iomanip>
#include <cuda_fp16.h>

#define checkCudaDrvErrors(call)                                                                                   \\
    do {                                                                                                           \\
        CUresult status = call;                                                                                    \\
        if (status != 0) {                                                                                         \\
            const char *msg = nullptr;                                                                             \\
            cuGetErrorString(status, &msg);                                                                        \\
            std::cerr << "CUDA error at line " << __LINE__ << " in file " << __FILE__ << ": " << msg << std::endl; \\
            exit(1);                                                                                               \\
        }                                                                                                          \\
    } while (0)

std::string ReadFileToString(const char *filename)
{
    std::ifstream ifile(filename);
    std::ostringstream buf;
    char ch;
    while (buf && ifile.get(ch))
        buf.put(ch);
    return buf.str();
}

extern "C" void cuda_runtime_profiling(
    rt_code_params_list,
    int number=1,
    int repeat=1,
    int min_repeat_ms=0
    )
{
    std::cout<< "Start runtime profiling:" << std::endl;
    // initialize cuda
    checkCudaDrvErrors(cuInit(0));
    // get the number of cuda devices
    int device_count = 0;
    checkCudaDrvErrors(cuDeviceGetCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No cuda devices found" << std::endl;
        exit(1);
    }

    // get the first cuda device
    CUdevice device;
    checkCudaDrvErrors(cuDeviceGet(&device, 0));

    // create a cuda context on the device
    CUcontext context;
    checkCudaDrvErrors(cuCtxCreate(&context, 0, device));

    CUstream stream;
    cuStreamCreate(&stream, 0);

    std::string ptx_code = ReadFileToString(rt_code_ptx_path);

    CUmodule module;
    checkCudaDrvErrors(cuModuleLoadData(&module, ptx_code.c_str()));

    // get the cuda kernel function
    CUfunction kernel;
    checkCudaDrvErrors(cuModuleGetFunction(&kernel, module, rt_code_kernel_name));

    // allocate input and output arrays on the gpu
rt_code_mem_alloc

    // copy input arrays from cpu to gpu
rt_code_mem_copy_htod

rt_code_set_grid_params
rt_code_set_block_params
rt_code_init_memref_params
    std::cout<< "Profiling init done;" << std::endl;
    // launch the kernel
    void *args[] = {rt_code_set_args_params};

    // skip first launch
    checkCudaDrvErrors(cuLaunchKernel(kernel, gx, gy, gz, bx, by, bz, 0, stream, args, NULL));
    cuStreamSynchronize(stream);
    std::vector<double> res;
    for (int i = 0; i < repeat; ++i) {

        std::chrono::time_point<
            std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;


        tbegin = std::chrono::high_resolution_clock::now();
        // start timing
        for (int i = 0; i < number; ++i) {
            checkCudaDrvErrors(cuLaunchKernel(kernel, gx, gy, gz, bx, by, bz, 0, stream, args, NULL));
        }
        cuStreamSynchronize(stream);
        tend = std::chrono::high_resolution_clock::now();

        // ns->ms
        double speed = std::chrono::duration_cast<std::chrono::duration<double> >(
            tend - tbegin).count() / number;
        res.push_back(speed);
    }

    // copy output array from gpu to cpu
rt_code_mem_copy_dtoh
    // free memory
rt_code_free_d_mem
    cuStreamDestroy(stream);
    cuCtxDestroy(context);
    double avg = 0.0;
    for (double r : res) {
        avg += r;
    }
    avg /= static_cast<double>(res.size());
    std::cout<< "average latency = " <<  std::fixed << std::setprecision(10)  << avg * 1000 << "ms" << std::endl;
    std::cout<< "Finish runtime profiling." << std::endl;
}


extern "C" void cuda_runtime_exec(rt_code_params_list)
{
    std::cout<< "Start runtime execution." << std::endl;
    // initialize cuda
    checkCudaDrvErrors(cuInit(0));

    // get the number of cuda devices
    int device_count = 0;
    checkCudaDrvErrors(cuDeviceGetCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No cuda devices found" << std::endl;
        exit(1);
    }

    // get the first cuda device
    CUdevice device;
    checkCudaDrvErrors(cuDeviceGet(&device, 0));

    // create a cuda context on the device
    CUcontext context;
    checkCudaDrvErrors(cuCtxCreate(&context, 0, device));

    std::string ptx_code = ReadFileToString(rt_code_ptx_path);

    CUmodule module;
    checkCudaDrvErrors(cuModuleLoadData(&module, ptx_code.c_str()));

    // get the cuda kernel function
    CUfunction kernel;
    checkCudaDrvErrors(cuModuleGetFunction(&kernel, module, rt_code_kernel_name));

    // allocate input and output arrays on the gpu
rt_code_mem_alloc

    // copy input arrays from cpu to gpu
rt_code_mem_copy_htod

rt_code_set_grid_params
rt_code_set_block_params
rt_code_init_memref_params

    // launch the kernel
    void *args[] = {rt_code_set_args_params};
    checkCudaDrvErrors(cuLaunchKernel(kernel, gx, gy, gz, bx, by, bz, 0, NULL, args, NULL));

    // copy output array from gpu to cpu
rt_code_mem_copy_dtoh
    // free memory
rt_code_free_d_mem
    std::cout<< "Finish runtime execution." << std::endl;
}
'''

cpu_profiling_template = '''
  llvm.func @nanoTime() -> i64 attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %0 = llvm.call @_mlir_ciface_nanoTime() : () -> i64
    llvm.return %0 : i64
  }
  llvm.func @warmUp(%arg0 : INPUTS_PTR) -> () attributes {llvm.emit_c_interface, sym_visibility = "private"} {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %c100 = llvm.mlir.constant(1000 : index) : i64
    llvm.br ^bb1(%c0 : i64)
  ^bb1(%2: i64):  // 2 preds: ^bb0, ^bb2
    %1 = llvm.icmp "slt" %2, %c100 : i64
    llvm.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @KERNEL_NAME(%arg0) : (INPUTS_PTR) -> ()
    %3 = llvm.add %2, %c1  : i64
    llvm.br ^bb1(%3 : i64)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @_mlir_ciface_nanoTime() -> i64 attributes {llvm.emit_c_interface, sym_visibility = "private"}
  llvm.func @main(INPUTS_NAME : INPUTS_PTR, %arg_time: !llvm.ptr<i64>) attributes {llvm.emit_c_interface, sym_visibility = "public"} {
    %c0 = llvm.mlir.constant(0 : index) : i64
    %c1 = llvm.mlir.constant(1 : index) : i64
    %ctimes = llvm.mlir.constant(CTIMES : index) : i64
    llvm.call @warmUp(INPUTS_NAME) : (INPUTS_PTR) -> ()
    %0 = llvm.call @nanoTime() : () -> i64
    llvm.br ^bb1(%c0 : i64)
  ^bb1(%2: i64):  // 2 preds: ^bb0, ^bb2
    %1 = llvm.icmp "slt" %2, %ctimes : i64
    llvm.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.call @KERNEL_NAME(INPUTS_NAME) : (INPUTS_PTR) -> ()
    %3 = llvm.add %2, %c1  : i64
    llvm.br ^bb1(%3 : i64)
  ^bb3:  // pred: ^bb1
    %4 = llvm.call @nanoTime() : () -> i64
    %5 = llvm.sub %4, %0  : i64
    llvm.store %5, %arg_time : !llvm.ptr<i64>
    llvm.return
  }
'''
