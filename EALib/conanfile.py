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
