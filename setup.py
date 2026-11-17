"""Build the Cython extensions against a statically linked copy of VLFeat.

VLFeat (BSD-2-Clause) is vendored as a pinned git submodule in extern/vlfeat and its C sources
are compiled once into a static ``vl`` library that every extension links against, so wheels
do not need a system libvl.
"""

import hashlib
import os
import platform
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext

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
    """build_clib that compiles in parallel and shares the result between Python versions.

    * distutils compiles serially; here the C sources are compiled in a thread pool.
    * libvl does not depend on Python, but every wheel build (one per Python version) would
      compile it again in its own build/temp.*-cpython-3xx directory. Instead it is built into
      build/libvl-<key>/, where the key hashes everything that influences the object code
      (compiler command line, macros, flags, VLFeat sources), so a later build for another Python
      version reuses it exactly when it would have produced an identical library.
    """

    def _key(self, build_info):
        h = hashlib.sha256()
        compiler = self.compiler
        command = (
            getattr(compiler, "compiler_so", None)  # unix
            or getattr(compiler, "compile_options", None)  # msvc
            or []
        )
        h.update(
            repr(
                (
                    sys.platform,
                    platform.machine(),
                    command,
                    getattr(compiler, "cc", None),
                    build_info.get("macros"),
                    build_info.get("cflags"),
                    build_info.get("include_dirs"),
                    os.environ.get("MACOSX_DEPLOYMENT_TARGET"),
                )
            ).encode()
        )
        for path in sorted((VLFEAT / "vl").iterdir()):
            if path.suffix in (".c", ".h", ".tc", ".th"):
                h.update(path.name.encode())
                h.update(path.read_bytes())
        return h.hexdigest()[:16]

    def build_libraries(self, libraries):
        for lib_name, build_info in libraries:
            # build_ext links against build_clib.build_clib, so point it at the shared directory
            shared = ROOT / "build" / f"{lib_name}-{self._key(build_info)}"
            self.build_clib = str(shared)
            library = self.compiler.library_filename(lib_name, output_dir=str(shared))
            if os.path.exists(library):
                print(f"reusing {library}")
                continue

            sources = list(build_info["sources"])
            chunks = [sources[i::JOBS] for i in range(JOBS) if sources[i::JOBS]]

            def compile_chunk(chunk, build_info=build_info):
                return self.compiler.compile(
                    chunk,
                    output_dir=str(shared / "obj"),
                    macros=build_info.get("macros"),
                    include_dirs=build_info.get("include_dirs"),
                    extra_postargs=build_info.get("cflags"),
                    debug=self.debug,
                )

            with ThreadPoolExecutor(len(chunks)) as pool:
                objects = [obj for objs in pool.map(compile_chunk, chunks) for obj in objs]
            self.compiler.create_static_lib(
                objects, lib_name, output_dir=str(shared), debug=self.debug
            )


def stable_numpy_include():
    """Copy numpy's headers to a fixed path, build/numpy-include-<version>.

    pip builds in an isolated environment with a random name (/tmp/build-env-xxxx), so numpy's
    real include directory differs on every build and so would the compile command line, which
    defeats compiler caches such as ccache. The headers are small and keyed by numpy version.
    """
    dest = ROOT / "build" / f"numpy-include-{np.__version__}"
    if not dest.exists():
        shutil.copytree(np.get_include(), dest)
    return str(dest)


class StableBuildExt(build_ext):
    """build_ext whose compile command line does not depend on the (random) build venv path."""

    def finalize_options(self):
        super().finalize_options()
        # distutils adds <build venv>/include; in pip's isolated build environment that directory
        # has no headers and only makes the command line vary from build to build
        self.include_dirs = [
            d
            for d in self.include_dirs
            if any((Path(d) / header).exists() for header in ("Python.h", "pyconfig.h"))
        ]


def extension(pyx):
    return Extension(
        name=pyx[:-4].replace("/", "."),
        sources=[pyx],
        include_dirs=[stable_numpy_include(), str(ROOT)] + include_dirs,
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
    cmdclass={"build_clib": ParallelBuildClib, "build_ext": StableBuildExt},
    options={"build_ext": {"parallel": JOBS}},
    ext_modules=cythonize(
        [extension(m) for m in modules],
        quiet=True,
        language_level="3",
        # always regenerate the C: it must match the Cython and numpy of the machine that compiles
        # it. Cython skips files whose .c is newer than the .pyx and its numpy .pxd, which is how a
        # stale .c from an sdist built against numpy 2.5 got compiled against numpy 2.4 headers.
        force=True,
        # no nthreads: cythonize uses multiprocessing, which re-runs setup.py in the workers
        # (spawn/forkserver is the default on macOS, Windows and Python 3.14)
    ),
)
