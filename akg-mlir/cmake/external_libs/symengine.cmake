if(ENABLE_GITEE)
    set(SYMENGINE_URL "https://gitee.com/mirrors/SymEngine/repository/archive/v0.11.2.tar.gz")
    set(SYMENGIN_MD5 "b7c94c609f5e634bd60a189e176e82eb")
else()
    set(SYMENGINE_URL "https://github.com/symengine/symengine/archive/refs/tags/v0.11.2.tar.gz")
    set(SYMENGIN_MD5 "4074f3c76570bdc2ae9914edafa29eb6")
endif()

akg_add_pkg(symengine
        VER 0.11.2
        LIBS symengine
        URL ${SYMENGINE_URL}
        MD5 ${SYMENGIN_MD5}
	CMAKE_OPTION -DHAVE_SYMENGINE_NOEXCEPT:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DWITH_BFD=OFF
    -DWITH_SYMENGINE_ASSERT:BOOL=OFF -DWITH_SYMENGINE_RCP:BOOL=ON -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF 
    -DWITH_SYMENGINE_THREAD_SAFE:BOOL=OFF -DWITH_ECM:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_BENCHMARKS:BOOL=OFF 
    -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON)
message("symengine_inc :${symengine_INC}")
include_directories(${symengine_INC})
add_library(akg::symengine ALIAS symengine::symengine)
