from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext

# 启用并行编译
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

ext_modules = [
    Pybind11Extension(
        "fastholo",
        sources=[
            "bindings/pybind_wrapper.cpp",
        ],
        libraries=['holo_recons_lib'],
        extra_compile_args=['-std=c++17', '-O3'],
        language='c++',
    ),
]

setup(
    name="fastholo",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
