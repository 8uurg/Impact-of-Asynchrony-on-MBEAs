# EAlib

## Installation

Requires a recent compiler that supports C++17 features, e.g.:
* GCC 9 or greater
* MSVC 19 or greater
* clang 10 or greater.

Older compilers may work, but have not been tested.

Build is done through meson. If not installed, install meson and ninja using `pip install meson ninja`.
Dependency on Cap'n Proto is a bit finicky, on Linux install using your package manager:
* apt: `apt-get install capnproto libcapnp-dev` 

On Windows recommended method is to build using the conan package manager:
``
    pip install conan
    mkdir conan
    conan install ..
    conan build ..    
``
Tip: make sure you read the error messages, you might need to build the dependencies.

### Setting up the build directory
*Debug:*
```
meson build --buildtype=debug
```

*Release:*
```
meson build --buildtype=release
```

### Compiling
*All targets:*
```
meson compile -C build
```

### Python wheels
While the aforementioned commands will compile a Python module, it won't build a wheel or install it using pip.
For this build (installed with `pip install build`) can be used:

```
python -m build .
```

### Documentation
Requires [`doxygen`](https://www.doxygen.nl/download.html), [`sphinx`](https://www.sphinx-doc.org), [`breathe`](https://github.com/michaeljones/breathe) and [`sphinx-rtd-theme`](https://pypi.org/project/sphinx-rtd-theme/).

Once set up, ensure that ensure that the build was configured to generate the documentation, e.g. using `meson configure -C build -Dbuild_doc=true` or providing the argument during the creation of the initial build directory. Generating/updating the documentation can be performed through `ninja -C build docs`.