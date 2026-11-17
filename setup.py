"""Build the Cython extensions against a statically linked copy of VLFeat.

VLFeat (BSD-2-Clause) is vendored as a pinned git submodule in extern/vlfeat and its C sources
are compiled once into a static ``vl`` library that every extension links against, so wheels
do not need a system libvl.
"""

import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_clib import build_clib

ROOT = Path(__file__).parent
VLFEAT = ROOT / "extern" / "vlfeat"
IS_WIN = sys.platform == "win32"
JOBS = int(os.environ.get("CYVLFEAT_BUILD_JOBS", min(os.cpu_count() or 1, 8)))
IS_X86 = platform.machine().lower() in ("x86_64", "amd64", "x86", "i386", "i686")

if not (VLFEAT / "vl" / "generic.c").exists():
    raise SystemExit(
        "extern/vlfeat is empty: run `git submodule update --init` (or build from an sdist)"
    )

vl_sources = sorted(
    str(p.relative_to(ROOT).as_posix())
    for p in (VLFEAT / "vl").glob("*.c")
    # getopt_long is only used by the VLFeat command line tools
    if p.name != "getopt_long.c"
)

# AVX is chosen at compile time only, so enabling it would crash on older CPUs; the SSE2
# paths exist only on x86; OpenMP is not used.
macros = [("VL_DISABLE_AVX", None), ("VL_DISABLE_OPENMP", None)]
# (MSVC never defines __SSE2__, which the SSE2 sources insist on, so they are off on Windows)
if not IS_X86 or IS_WIN:
    macros.append(("VL_DISABLE_SSE2", None))
if IS_WIN:
    # a static library: have MSVC emit plain symbols instead of dllimport stubs
    macros += [("VL_BUILD_DLL", None), ("_CRT_SECURE_NO_WARNINGS", None)]

include_dirs = [str(VLFEAT)]
warning_flags = [] if IS_WIN else ["-Wno-unused-function", "-Wno-implicit-function-declaration"]

libvl = (
    "vl",
    {
        "sources": vl_sources,
        "include_dirs": include_dirs,
        "macros": macros,
        "cflags": warning_flags + ([] if IS_WIN else ["-fvisibility=hidden"]),
    },
)


class ParallelBuildClib(build_clib):
    """build_clib that compiles the C sources of a library in parallel (distutils is serial)."""

    def build_libraries(self, libraries):
        for lib_name, build_info in libraries:
            sources = list(build_info["sources"])
            chunks = [sources[i::JOBS] for i in range(JOBS) if sources[i::JOBS]]

            def compile_chunk(chunk, build_info=build_info):
                return self.compiler.compile(
                    chunk,
                    output_dir=self.build_temp,
                    macros=build_info.get("macros"),
                    include_dirs=build_info.get("include_dirs"),
                    extra_postargs=build_info.get("cflags"),
                    debug=self.debug,
                )

            with ThreadPoolExecutor(len(chunks)) as pool:
                objects = [obj for objs in pool.map(compile_chunk, chunks) for obj in objs]
            self.compiler.create_static_lib(
                objects, lib_name, output_dir=self.build_clib, debug=self.debug
            )


def extension(pyx):
    return Extension(
        name=pyx[:-4].replace("/", "."),
        sources=[pyx],
        include_dirs=[np.get_include(), str(ROOT)] + include_dirs,
        define_macros=macros + [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=["vl"],
        extra_compile_args=warning_flags,
        language="c",
    )


modules = [
    "cyvlfeat/fisher/cyfisher.pyx",
    "cyvlfeat/generic/generic.pyx",
    "cyvlfeat/gmm/cygmm.pyx",
    "cyvlfeat/hog/cyhog.pyx",
    "cyvlfeat/kmeans/cykmeans.pyx",
    "cyvlfeat/quickshift/cyquickshift.pyx",
    "cyvlfeat/sift/cysift.pyx",
    "cyvlfeat/vlad/cyvlad.pyx",
]

setup(
    libraries=[libvl],
    cmdclass={"build_clib": ParallelBuildClib},
    options={"build_ext": {"parallel": JOBS}},
    ext_modules=cythonize(
        [extension(m) for m in modules],
        quiet=True,
        language_level="3",
        # no nthreads: cythonize uses multiprocessing, which re-runs setup.py in the workers
        # (spawn/forkserver is the default on macOS, Windows and Python 3.14)
    ),
)
