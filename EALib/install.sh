cd $(dirname $(readlink -f "$0")) >/dev/null

meson setup build_ur --unity=on --buildtype=release -D python.install_env=auto
meson setup build_ur --reconfigure --unity=on --buildtype=release -D python.install_env=auto
meson install -C build_ur
