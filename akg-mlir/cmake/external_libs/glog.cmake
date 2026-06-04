set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS}")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/glog/repository/archive/v0.7.1.tar.gz")
    set(SHA256 "54854d52a4a0f12a7a57f43d22457477281ef373b6487c5ac422e6303d7ff3e8")
else()
    set(REQ_URL "https://github.com/google/glog/archive/v0.7.1.tar.gz")
    set(SHA256 "00e4a87e87b7e7612f519a41e491f16623b12423620006f59f5688bfd8d13b08")
endif()

set(glog_option -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF -DWITH_GFLAGS=OFF -DCMAKE_BUILD_TYPE=Release)

akg_add_pkg(glog
    VER 0.7.1
    LIBS glog
    URL ${REQ_URL}
    SHA256 ${SHA256}
    CMAKE_OPTION ${glog_option}
    CUSTOM_CMAKE_GENERATOR Ninja)
include_directories(${glog_INC})
add_library(glog ALIAS glog::glog)
