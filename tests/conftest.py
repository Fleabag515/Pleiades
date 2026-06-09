import os
import tempfile

# Isolate Pleiades home to a temp dir BEFORE importing the package.
_tmp = tempfile.mkdtemp(prefix="pleiades-test-")
os.environ["PLEIADES_HOME"] = _tmp
os.environ.setdefault("PLEIADES_MASTER_KEY", "")
