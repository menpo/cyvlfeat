execute:
	CFLAGS="-I${CONDA_PREFIX}/include ${CFLAGS}" LDFLAGS="-L${CONDA_PREFIX}/lib ${LDFLAGS}"  python setup.py build_ext --inplace
