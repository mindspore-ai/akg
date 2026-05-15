// DenseNet121-style GAP f16: mean.dim lowers to sum+div via f32 in the full mfusion pipeline.
// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize \
// RUN:   --decompose="pattern-type=AFTER_MANUAL_FUSION" --canonicalize \
// RUN:   --convert-mfuse-to-torch="kernel-generator=dvm" --reconcile-unrealized-casts --canonicalize | FileCheck %s

module {
  func.func @gap_head(%x: !torch.vtensor<[4,1024,7,7],f16>) -> !torch.vtensor<[4,1024,1,1],f16> {
    %none = torch.constant.none
    %true = torch.constant.bool true
    %c2 = torch.constant.int 2
    %c3 = torch.constant.int 3
    %dims = torch.prim.ListConstruct %c2, %c3 : (!torch.int, !torch.int) -> !torch.list<int>
    %act = torch.aten.relu %x : !torch.vtensor<[4,1024,7,7],f16> -> !torch.vtensor<[4,1024,7,7],f16>
    %pool = torch.aten.mean.dim %act, %dims, %true, %none
        : !torch.vtensor<[4,1024,7,7],f16>, !torch.list<int>, !torch.bool, !torch.none
        -> !torch.vtensor<[4,1024,1,1],f16>
    return %pool : !torch.vtensor<[4,1024,1,1],f16>
  }
}

// CHECK-LABEL: func.func @gap_head
// CHECK-NOT: torch.aten.mean.dim
// CHECK-DAG: torch.constant.int 49
// CHECK-DAG: torch.prims.convert_element_type
// CHECK-DAG: torch.aten.sum.dim_IntList
// CHECK-DAG: torch.aten.div.Scalar
