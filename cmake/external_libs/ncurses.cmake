
set(NCURSES_URL "https://github.com/mirror/ncurses/archive/refs/tags/v6.2.tar.gz")
set(NCURSES_MD5 "71b9c4fed2aa948cbac051f3cccfcfb6")

akg_add_pkg(ncurses
        VER 6.2
        LIBS ncurses
        URL ${NCURSES_URL}
        MD5 ${NCURSES_MD5}
        CONFIGURE_COMMAND ./configure CFLAGS=-fPIC CPPFLAGS=-fPIC)
link_directories("${ncurses_LIBPATH}")
add_library(akg::ncurses ALIAS ncurses::ncurses)
