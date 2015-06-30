import nose
import sys


if nose.run(argv=['', 'cyvlfeat']):
    sys.exit(0)
else:
    sys.exit(1)
