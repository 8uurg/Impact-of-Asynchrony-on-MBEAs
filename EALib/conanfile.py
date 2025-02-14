#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from conans import ConanFile, Meson, tools


class EALibConan(ConanFile):
    generators = "pkg_config"
    name = "EALib"
    version = "0.0.1"
    # thrift/0.16.0, capnproto/0.9.1@
    requires = ["cmake/3.23.2@", "thrift/0.16.0@"]
    settings = "os", "compiler", "build_type", "arch"

    def build(self):
        meson = Meson(self)
        meson.configure(build_folder="build")
        meson.build()
