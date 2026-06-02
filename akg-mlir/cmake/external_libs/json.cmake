set(nlohmann_json_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(nlohmann_json_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/JSON-for-Modern-CPP/repository/archive/v3.12.0.zip")
    set(SHA256 "ed098b2314ebd9b92cc4113bbdec15d570be2db1d7d8a796493432df115b4821")
    set(INCLUDE "./include")
else()
    ## The nlohmann_json naming convention is used to distinguish different versions of the include.zip file.
    set(REQ_URL "https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip")
    set(SHA256 "b8cb0ef2dd7f57f18933997c9934bb1fa962594f701cd5a8d3c2c80541559372")
    set(INCLUDE "./include")
endif()

akg_add_pkg(nlohmann_json
    VER 3.12.0
    HEAD_ONLY ${INCLUDE}
    URL ${REQ_URL}
    SHA256 ${SHA256}
    CUSTOM_CMAKE_GENERATOR Ninja)
include_directories(${nlohmann_json_INC})
add_library(akg::nlohmann_json ALIAS nlohmann_json)
