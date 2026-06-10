#!/usr/bin/env python3
"""
Launcher script for the Customer Segmentation Dashboard.
Handles Windows ProactorEventLoop connection reset errors gracefully.
"""

import sys
import os
import asyncio

# Fix for Windows + asyncio + ProactorEventLoop connection reset errors
# This must be applied before Streamlit imports
if sys.platform == "win32":
    from asyncio.proactor_events import _ProactorBasePipeTransport
    from functools import wraps

    def silence_connection_reset_error(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except ConnectionResetError:
                pass
        return wrapper

    _ProactorBasePipeTransport.__del__ = silence_connection_reset_error(_ProactorBasePipeTransport.__del__)
    _ProactorBasePipeTransport._call_connection_lost = silence_connection_reset_error(_ProactorBasePipeTransport._call_connection_lost)


if __name__ == "__main__":
    os.chdir("C:\\Users\\Admin\\Documents\\Projects\\Python Projects\\Customer Segmentation Dashboard")
    import subprocess

    print("Starting Customer Segmentation Dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501",
        "--server.headless=true"
    ])
