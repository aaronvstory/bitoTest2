# enhanced_ui.py
import threading
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.spinner import Spinner
from datetime import datetime
from typing import List


class EnhancedUI:
    def __init__(self):
        self.console = Console()
        self._lock = threading.Lock()

    def status(self, message: str, level: str = "info", transient: bool = False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        with self._lock:
            style_map = {"info": "blue", "success": "green", "error": "red", "warning": "yellow", "debug": "dim white"}
            style = style_map.get(level, "white")
            msg = f"[{timestamp}] [{style}]{message}[/{style}]"
            if transient:
                self.console.print(msg, end="\r")
            else:
                self.console.print(msg)

    def show_spinner(self, message: str):
        """Show a spinner with message"""
        spinner = Spinner("dots")
        with Live(spinner, refresh_per_second=10) as live:
            while True:
                live.update(f"{spinner} {message}")
                yield
