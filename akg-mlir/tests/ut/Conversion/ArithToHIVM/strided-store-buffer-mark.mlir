// RUN: akg-opt %s -convert-arith-to-hivm | FileCheck %s
// The GM write is strided although the read is contiguous. CANN 9.1
// requires padding marks on the whole producer chain, including casts.
// CHECK-LABEL: func.func @strided_store
// CHECK: buffer_size_in_byte = 544
// CHECK: buffer_size_in_byte = 544
// CHECK: buffer_size_in_byte = 544
func.func @strided_store(%src: memref<?xbf16>, %dst: memref<?xbf16, strided<[640], offset: ?>>, %n: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %max = arith.constant 17 : index
  %pad = arith.constant 0.0 : bf16
  %v = npuvector.transfer_read %src[%c0][%n][%max], %pad : memref<?xbf16>, !npuvector<?xbf16>
  %f = npuvector.extf %v : !npuvector<?xbf16> to !npuvector<?xf32>
  %b = npuvector.truncf %f : !npuvector<?xf32> to !npuvector<?xbf16>
  npuvector.transfer_write %b, %dst[%c0] : !npuvector<?xbf16>, memref<?xbf16, strided<[640], offset: ?>>
  return
}
// A contiguous store must not acquire a padded trailing axis.
// CHECK-LABEL: func.func @contiguous_store
// CHECK: buffer_size_in_byte = 64
// CHECK-NOT: buffer_size_in_byte = 544
func.func @contiguous_store(%src: memref<?xbf16>, %dst: memref<?xbf16>, %n: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %max = arith.constant 17 : index
  %pad = arith.constant 0.0 : bf16
  %v = npuvector.transfer_read %src[%c0][%n][%max], %pad : memref<?xbf16>, !npuvector<?xbf16>
  npuvector.transfer_write %v, %dst[%c0] : !npuvector<?xbf16>, memref<?xbf16>
  return
}
