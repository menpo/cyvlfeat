#!/bin/sh

cp -r $RECIPE_DIR/../ .

if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  export CFLAGS="-I$PREFIX/include $CFLAGS"
  export LDFLAGS="-L$PREFIX/lib $LDFLAGS"
fi
if [ "$(uname -s)" == "Darwin" ]; then
  export CFLAGS="-I$PREFIX/include -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET $CFLAGS"
  export LDFLAGS="-L$PREFIX/lib -Wl,-headerpad_max_install_names $LDFLAGS"
fi

"$PYTHON" setup.py install --single-version-externally-managed --record=/tmp/record.txt

if [ "$(uname -s)" == "Darwin" ]; then
  # For some reason Cython refuses to generate @rpath/vlfeat.dylib - so
  # we manually do it here
  find $SP_DIR/cyvlfeat -name "*.so" -print0 | while read -d $'\0' file
  do
    install_name_tool -change @loader_path/libvl.dylib @rpath/libvl.dylib $file
  done
fi

# Build the pypi package
"$PYTHON" setup.py sdist
