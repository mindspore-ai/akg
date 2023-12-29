module @template_matmul_f32_f32_f32 {
  func.func @template_matmul_f32_f32_f32(%targ0: tensor<?x?xf32>, %targ1: tensor<?x?xf32>, %targ2: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf32>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf32>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5] : memref<?x?xf32>, f32
          fusion.multi_load %arg0[%arg3, %arg5] : memref<?x?xf32>, f32
          %4 = fusion.load %arg1[%arg5, %arg4] : memref<?x?xf32>, f32
          fusion.multi_load %arg1[%arg5, %arg4] : memref<?x?xf32>, f32
          %5 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
          %30 = fusion.insert %arg0, %3 : memref<?x?xf32>, f32 to f32
          %40 = fusion.insert %arg1, %4 : memref<?x?xf32>, f32 to f32
          %6 = arith.mulf %30, %40 : f32
          %7 = arith.addf %5, %6 : f32
          %last = arith.subi %k, %c1 : index
          %is_last = arith.cmpi eq, %arg5, %last : index
          %700 = scf.if %is_last -> f32 {
            fusion.multi_load %arg2[%arg3, %arg4] : memref<?x?xf32>, f32
            %70 = fusion.insert %arg2, %7 : memref<?x?xf32>, f32 to f32
            scf.yield %70 : f32
          } else {
            scf.yield %7 : f32
          }
          fusion.store %700, %arg2[%arg3, %arg4] : f32, memref<?x?xf32>
        }
      }
    }
    return
  }
}
