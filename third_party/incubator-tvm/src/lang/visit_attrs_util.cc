#include <tvm/operation.h>
#include <tvm/ir.h>
#include <tvm/visit_attrs_util.h>
namespace air {

void VisitMap(const Map<std::string, NodeRef>& source_map, Map<String, NodeRef>& dist_map) {
  for (auto& item : source_map) {
    auto key = item.first;
    auto node = item.second;
    if(dist_map.count(key) && dist_map[key].same_as(node)){
      continue;
    }
    String str = key;
    dist_map.Set(str, node);
  }
}

}  // namespace air