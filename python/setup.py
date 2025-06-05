from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile, build_ext
import os

# 启用并行编译
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

# 项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# 简化的扩展模块定义
ext_modules = [
    Pybind11Extension(
        "fastholo",
        sources=[
            "bindings/pybind_wrapper.cpp",
        ],
        include_dirs=[
            os.path.join(project_root, 'include'),
            '/usr/local/cuda/include',
            '/usr/local/include/opencv4',
            '/usr/include'
        ],
        library_dirs=[
            '/usr/local/cuda/lib64',
            '/usr/lib/x86_64-linux-gnu',
            '/usr/local/lib'
        ],
        libraries=[
            'holo_recons_lib'  # 我们将创建一个预编译的库
        ],
        extra_compile_args=[
            '-std=c++17',
            '-O3'
        ],
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
