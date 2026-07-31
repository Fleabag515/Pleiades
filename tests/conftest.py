import atexit
import os
import shutil
import tempfile

# Isolate Pleiades home to a temp dir BEFORE importing the package.
_tmp = tempfile.mkdtemp(prefix="pleiades-test-")
os.environ["PLEIADES_HOME"] = _tmp
os.environ.setdefault("PLEIADES_MASTER_KEY", "")

# This runs at collection time (module scope, before any fixture machinery
# exists), so a normal fixture teardown can't reach it -- atexit is the only
# hook available. Without this, every pytest invocation leaks a fresh
# pleiades-test-* directory (profiles, vault DBs, scheduled_tasks.json, etc.)
# into the OS temp dir forever.
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)
