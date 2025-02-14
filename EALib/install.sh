#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

cd $(dirname $(readlink -f "$0")) >/dev/null

meson setup build_ur --unity=on --buildtype=release -D python.install_env=auto
meson setup build_ur --reconfigure --unity=on --buildtype=release -D python.install_env=auto
meson install -C build_ur
