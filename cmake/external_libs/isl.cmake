set(isl_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(isl_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(isl
        VER 0.22
        URL http://isl.gforge.inria.fr/isl-0.22.tar.gz
        MD5 671d0a5e10467a5c6db0893255278845
        PATCHES ${CMAKE_CURRENT_SOURCE_DIR}/third_party/patch/isl/isl.patch)
