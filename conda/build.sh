#!/bin/sh

cp -r $RECIPE_DIR/../ .

export CFLAGS="-I$PREFIX/include $CFLAGS"
export LDFLAGS="-L$PREFIX/lib $LDFLAGS"

"$PYTHON" setup.py install --single-version-externally-managed --record=/tmp/record.txt

