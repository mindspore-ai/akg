module @template_batch_matmul_f16_f16_f16 {
  func.func @template_batch_matmul_f16_f16_f16(%targ0: tensor<?x?x?xf16>, %targ1: tensor<?x?x?xf16>, %targ2: tensor<?x?x?xf16> {bufferization.access = "write"}, %batch: index, %m: index, %n:index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?x?xf16>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?x?xf16>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?x?xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %batch step %c1 {
      scf.for %arg4 = %c0 to %m step %c1 {
        scf.for %arg5 = %c0 to %n step %c1 {
          scf.for %arg6 = %c0 to %k step %c1 {
            %4 = fusion.load %arg0[%arg3, %arg4, %arg6] : memref<?x?x?xf16>, f16
            fusion.multi_load %arg0[%arg3, %arg4, %arg6] : memref<?x?x?xf16>, f16
            %5 = fusion.load %arg1[%arg3, %arg6, %arg5] : memref<?x?x?xf16>, f16
            fusion.multi_load %arg1[%arg3, %arg6, %arg5] : memref<?x?x?xf16>, f16
            %6 = memref.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf16>
            %40 = fusion.insert %arg0, %4 : memref<?x?x?xf16>, f16 to f16
            %50 = fusion.insert %arg1, %5 : memref<?x?x?xf16>, f16 to f16
            %7 = arith.mulf %40, %50 : f16
            %8 = arith.addf %6, %7 : f16
            memref.store %8, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf16>
          }
        }
      }
    }
    return
  }
}
