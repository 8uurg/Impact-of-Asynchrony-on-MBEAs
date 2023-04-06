project('EAlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

ssmod = import('sourceset')
pythonmod = import('python')

python = pythonmod.find_installation('python3')

# Dependencies
dep_catch2 = dependency('catch2')
dep_trompeloeil = dependency('trompeloeil', fallback: ['trompeloeil', 'trompeloeil_dep'])
dep_python_embed = python.dependency(embed: true)
dep_python = python.dependency(embed: false)
dep_pybind11 = dependency('pybind11')
dep_cereal = dependency('cereal')
# dep_capnp = dependency('capnp')
# capnproto_compiler = find_program('capnp')

stacktrace = false

default_config = configuration_data({
  'test': false,
  'pythonembed': true,
  'pythonlib': false,
  'python': true,
})

# General source files
ss = ssmod.source_set()
include_dir = include_directories('include')

# Serialization & Deserialization.
ss.add(dep_cereal)

ss.add(files([
  'src/base.cpp',
  'src/logging.cpp',
  'src/acceptation_criteria.cpp',
  'src/initializers.cpp',
  'src/ga.cpp',
  'src/problems.cpp',
  'src/gomea.cpp',
  'src/running.cpp',
  'src/archive.cpp',
  'src/sim.cpp',
  'src/sim-gomea.cpp',
  'src/ecga.cpp',
  'src/sim-ga.cpp',
  'src/kernels.cpp',
  'src/debughelpers.cpp',
]))

# Add stacktrace requirements to the build, the configuration (when not commented out)
# uses backtrace, which is a GCC compiler feature, and hence does not work everywhere.
# Set the variable above to true, and uncomment the required includes to allow boosts'
# stacktrace functionality to work.
if (stacktrace)
  cxx = meson.get_compiler('cpp')
  libdl_dep = cxx.find_library('dl')
  backtrace_dep = cxx.find_library('backtrace')
  boost_dep = dependency('boost')
  traceback_deps = [libdl_dep, backtrace_dep, boost_dep]
  
  ss.add(traceback_deps)
endif

# Python sources & dependencies
ss.add(when: 'pythonlib', if_true: files([
  'src/python.cpp',
  'src/python/acceptation_criteria.cpp',
  'src/python/archive.cpp',
  'src/python/base.cpp',
  'src/python/ga.cpp',
  'src/python/gomea.cpp',
  'src/python/initializers.cpp',
  'src/python/problems.cpp',
  'src/python/logging.cpp',
  'src/python/running.cpp',
  'src/python/utilities.cpp',
  'src/python/sim.cpp',
  'src/python/sim-gomea.cpp',
  'src/python/sim-ga.cpp',
  'src/python/ecga.cpp',
]))
ss.add(when: 'pythonlib', if_true: dep_python)
ss.add(when: 'pythonembed', if_true: dep_python_embed)
ss.add(when: 'python', if_true: dep_pybind11)
# ss.add(dep_capnp)

# Test sources & dependencies
ss.add(when: 'test', if_true: files([
  'test/main.cpp',
  'test/test_base.cpp',
  'test/test_acceptation_criteria.cpp',
  'test/test_initializers.cpp',
  'test/test_utilities.cpp',
  'test/test_ga.cpp',
  'test/test_problems.cpp',
  'test/test_gomea.cpp',
  'test/test_running.cpp',
  'test/test_archive.cpp',
  'test/test_sim.cpp',
  'test/test_sim-gomea.cpp',
  'test/test_ecga.cpp',
  'test/test_sim_ga.cpp',
]))
ss.add(when: 'test', if_true: [dep_catch2, dep_trompeloeil])

# Python targets
python_config = configuration_data()
python_config.merge_from(default_config)
python_config.set('pythonlib', true)
python_config.set('pythonembed', false)

ss_python = ss.apply(python_config)
python.extension_module(
  'ealib',
  ss_python.sources(),
  include_directories: include_dir,
  dependencies: ss_python.dependencies(),
  install: true,
)

# Testing targets
testing_config = configuration_data()
testing_config.merge_from(default_config)
testing_config.set('test', true)

ss_test = ss.apply(testing_config)
test('EAlib tests', executable(
  'test_ealib',
  ss_test.sources(),
  include_directories: include_dir,
  dependencies: ss_test.dependencies(),
  install: false
))


if get_option('build_doc')
  subdir('doc')
endif