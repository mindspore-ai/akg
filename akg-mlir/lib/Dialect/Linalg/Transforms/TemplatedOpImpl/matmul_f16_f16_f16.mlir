module @template_matmul_f16_f16_f16 {
  func.func @template_matmul_f16_f16_f16(%targ0: tensor<?x?xf16>, %targ1: tensor<?x?xf16>, %targ2: tensor<?x?xf16> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf16>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf16>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5] : memref<?x?xf16>, f16
          fusion.multi_load %arg0[%arg3, %arg5] : memref<?x?xf16>, f16
          %4 = memref.load %arg1[%arg5, %arg4] : memref<?x?xf16>
          %5 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf16>
          %30 = fusion.insert %arg0, %3 : memref<?x?xf16>, f16 to f16
          %6 = arith.mulf %30, %4 : f16
          %7 = arith.addf %5, %6 : f16
          memref.store %7, %arg2[%arg3, %arg4] : memref<?x?xf16>
        }
      }
    }
    return
  }
}
