"""Launcher for the Streamlit UI.

Exposed as the ``vbm-ui`` console script and as ``python -m vbmitigator.ui``.
It shells out to ``streamlit run`` on :mod:`vbmitigator.ui.app`, forwarding any
extra CLI arguments to streamlit.

The subprocess form (``python -m streamlit run ...``) is used because it is the
version-independent, officially supported way to start a Streamlit app.

Remote-friendly defaults
------------------------
When running on a remote server (e.g. VS Code Remote / SSH) with the port
forwarded to your laptop, plain Streamlit often shows a **blank page** because:

* it tries to open a browser on the *remote* host,
* its XSRF/CORS defaults reject the websocket through the forwarding proxy, and
* its file-watcher can hit the remote's inotify limit and crash.

So unless you override them, we pass:

    --server.headless true
    --server.enableCORS false
    --server.enableXsrfProtection false
    --server.fileWatcherType none
    --browser.gatherUsageStats false

Any flag you pass on the command line takes precedence over these defaults.
"""

import importlib.util
import os
import subprocess
import sys

_DEFAULTS = {
    "--server.headless": "true",
    "--server.enableCORS": "false",
    "--server.enableXsrfProtection": "false",
    "--server.fileWatcherType": "none",
    "--browser.gatherUsageStats": "false",
}


def _apply_defaults(argv):
    """Return ``argv`` plus any default flag the user didn't already set."""
    present = {a.split("=", 1)[0] for a in argv}
    out = list(argv)
    for flag, value in _DEFAULTS.items():
        if flag not in present:
            out += [flag, value]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    if importlib.util.find_spec("streamlit") is None:
        sys.exit(
            "Streamlit is not installed. Install the UI extra, e.g.:\n"
            "    pip install -e '.[ui]'\n"
            "    # or\n"
            "    pip install streamlit"
        )

    args = _apply_defaults(argv)
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, *args]

    print(
        "Starting VB-Mitigator UI (headless).\n"
        "On VS Code Remote / SSH: open the forwarded port from the 'Ports' panel "
        "(or the http://localhost:<port> link Streamlit prints below) in your "
        "local browser.\n",
        flush=True,
    )
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
