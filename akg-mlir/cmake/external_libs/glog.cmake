set(glog_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 ${SECURE_CXX_FLAGS}")
set(glog_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

set(glog_CXXFLAGS "${glog_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(GLOG_URL "https://gitee.com/mirrors/glog/repository/archive/v0.4.0.tar.gz")
set(GLOG_MD5 "9a7598a00c569a11ff1a419076de4ed7")

set(glog_option -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON 
        -DBUILD_SHARED_LIBS=OFF -DWITH_GFLAGS=OFF -DCMAKE_BUILD_TYPE=Release)

akg_add_pkg(glog
        VER 0.4.0
        LIBS glog
        URL ${GLOG_URL}
        MD5 ${GLOG_MD5}
    CMAKE_OPTION ${glog_option})
include_directories(${glog_INC})
add_library(glog ALIAS glog::glog)