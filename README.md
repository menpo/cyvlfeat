cyvlfeat
========
A Python (cython) wrapper of the VLFeat library

We intend for this to be a light wrapper around the [VLFeat](http://www.vlfeat.org/) toolbox. cyvlfeat will provide a mixture of pure Python and Cython code that looks to replicate the existing Matlab toolbox. Cython is intended to fulfill the role of ``mex`` files.

We respect the original BSD 2-clause license and thus release this wrapper under the same license.

We thank the authors of VLFeat for their contribution to the computer vision community.

Current State
-------------
At the moment, the following methods from vlfeat are exposed:


  - fisher
    - fisher
  - generic
    - set_simd_enabled, get_simd_enabled, cpu_has_avx, cpu_has_sse3, cpu_has_sse2
    - get_num_cpus,
    - get_max_threads, set_num_threads, get_thread_limit
  - hog
    - hog
  - kmeans
    - kmeans
    - kmeans_quantize
    - ikmeans, ikmeans_push
    - hikmeans, hikmeans_push
  - sift
    - dsift
    - sift
    - phow

Installation
------------
Wheels with VLFeat bundled are published to PyPI for CPython 3.10-3.14 on Linux (x86_64 and aarch64,
glibc and musl), macOS (arm64 and x86_64) and Windows, so no compiler, conda or system ``vlfeat`` is needed:

    pip install cyvlfeat

or, from conda-forge:

    conda install -c conda-forge cyvlfeat

Development
-----------
To develop cyvlfeat (to extend its functionality), you will need to be comfortable using Cython. To begin re-implementing Matlab's ``mex`` methods in Cython you need a C compiler and [uv](https://docs.astral.sh/uv/) (or any Python 3.10+ environment). VLFeat itself is vendored as a git submodule (``extern/vlfeat``, a [fork](https://github.com/menpo/vlfeat) of VLFeat 0.9.21 with a one-line fix) and is compiled into a static library that every extension links, so there is nothing else to install.

To begin developing, you will need to git fork and clone this repository, including the submodule:

  - Fork using the Github fork button
  - Clone your repo:  ``git clone --recurse-submodules git@github.com:YOUR_GITHUB_USERNAME/cyvlfeat.git``
  - Add this repo as **upstream**:  ``git remote add upstream git@github.com:menpo/cyvlfeat.git``

You can now build cyvlfeat in a virtual environment and run the tests with:

    uv sync
    uv run pytest

(The tests download two sample images through ``scipy.datasets`` on first use.) Builds compile in parallel; set ``CYVLFEAT_BUILD_JOBS`` to change the number of jobs.

To add a new feature, please start a pull request. This kicks off the continuous integration, which builds and tests cyvlfeat on Linux (x86_64 and aarch64), macOS and Windows, and the full set of wheels.

Releasing
---------
Versions come from git tags (via ``setuptools-scm``). To release, tag the commit on ``master``
(``git tag -a v0.8.1 -m "cyvlfeat 0.8.1" && git push origin v0.8.1``) and approve the ``pypi`` deployment on the
resulting [Release workflow](https://github.com/menpo/cyvlfeat/actions/workflows/release.yml) run. The workflow
builds the wheels and sdist, uploads them to PyPI with trusted publishing (no tokens) and creates a GitHub
release (tags like ``v0.8.1rc1`` or ``v0.8.1.dev1`` are published as pre-releases). A manual run of the workflow
with ``publish: testpypi`` is a dry run. Do not put two release tags on the same commit: the version is derived
from the tag and the workflow refuses to publish if it does not match the tag. Tag builds never use the compiler cache.
