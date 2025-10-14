// RUN: akg-opt %s -split-input-file -akg-affine-loop-tile="tile-size=2" -allow-unregistered-dialect | FileCheck %s
// UNSUPPORTED: true 
// CHECK-DAG: [[$ID:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[$UB:#map[0-9]*]] = affine_map<(d0) -> (d0 + 64)>
// CHECK-DAG: [[$UB_INTRA_TILE:#map[0-9]*]] = affine_map<(d0) -> (d0 + 2)>
 
// CHECK-LABEL: func @loop_tiling()
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 256 step 64 {
// CHECK-NEXT:     affine.for %[[I:.*]] = [[$ID]](%{{.*}}) to [[$UB]](%{{.*}}) step 2 {
// CHECK-NEXT:       affine.for %[[J:.*]] = [[$ID]](%{{.*}}) to [[$UB_INTRA_TILE]](%{{.*}}) {
// CHECK-NEXT:          "test.foo"(%[[J]])
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:  return
 
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 32)>
func.func @loop_tiling() {
	affine.for %arg0 = 0 to 256 step 32 {
		affine.for %arg1 = #map(%arg0) to #map1(%arg0){
			"test.foo"(%arg1) : (index) -> ()
		}
	}
	return
}