"""Built-in tools. Import this package to register all tools into the global registry."""

from . import files      # noqa: F401  read/write/append/edit/preview/grep/symbols/list/find
from . import shell      # noqa: F401  run_shell / run_python
from . import web        # noqa: F401  web_search / http_fetch / deep_research
from . import git        # noqa: F401  git_status/diff/log/blame/show/branches/add/commit/checkout
from . import process    # noqa: F401  start_process / read_process_output / kill / send_stdin
from . import system     # noqa: F401  notify / context_status / clipboard / get_env / open_file
from . import memory     # noqa: F401  set_section / add_finding / note_to_self / remember / recall / forget
from . import documents  # noqa: F401  read_pdf / read_docx / read_xlsx / read_image
from . import quality    # noqa: F401  format_code / lint_code / run_tests
from . import events     # noqa: F401  start/stop_webhook_listener / wait_for_event / list_events
from . import browser    # noqa: F401  browser_open/read/click/fill/screenshot/close (optional dep)
from . import inference  # noqa: F401  hardware / models / profile / characters
from .. import subagent   # noqa: F401  dispatch_subagent / dispatch_subagents_parallel
from .. import toolsearch # noqa: F401  find_tools / list_catalog / call_tool
