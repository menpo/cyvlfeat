#!/bin/sh

cp -r $RECIPE_DIR/../ .

case "$OSTYPE" in
  linux*)   export CFLAGS="$CFLAGS -m${ARCH}";export LDLAGS="$LDLAGS -m${ARCH}";;
  *)        echo "WARNING: Unknown OS - ${OSTYPE}";;
esac

export CFLAGS="-I$PREFIX/include $CFLAGS"
export LDFLAGS="-L$PREFIX/lib $LDFLAGS"

"$PYTHON" setup.py install --single-version-externally-managed --record=/tmp/record.txt

