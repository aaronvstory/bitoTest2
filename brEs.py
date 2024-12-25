import os
from enhanced_UI import EnhancedUI
import sys
import time
import threading
import platform
import logging
import msvcrt
import json
import asyncio
import pickle
import re
import random
from datetime import datetime
from typing import Dict, List, Optional, Union, Set, Tuple, Any
from functools import wraps
from collections import deque, defaultdict
from dataclasses import dataclass, field

# Selenium imports
import undetected_chromedriver as uc
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
# Selenium imports
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException
)

# Rich for UI
from rich.console import Console
from rich.panel import Panel

# These should be defined after imports
exit_flag = threading.Event()
PRINT_LOCK = threading.Lock()
stats = {"sent": 0, "errors": 0, "checked": 0}


# AppState class definition (instead of importing from types)
@dataclass
class AppState:
    running: bool = True
    paused: bool = False
    driver: Optional[WebDriver] = None
    console: Console = field(default_factory=Console)
    # Add any other state variables you need


class UI:
    def __init__(self):
        self.console = Console()
        self._lock = threading.Lock()
        self.last_status = ""

    def status(self, message: str, level: str = "info") -> None:
        """Display a status message with proper formatting and threading safety"""
        with self._lock:
            style_map = {"info": "blue", "success": "green", "error": "red", "warning": "yellow"}
            style = style_map.get(level, "white")
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.print(f"[{timestamp}] [{style}]{message}[/{style}]")
            self.last_status = message

    def progress(self, current: int, total: int, message: str = "") -> None:
        """Display a progress bar"""
        with self._lock:
            percentage = (current / total) * 100 if total > 0 else 0
            self.console.print(f"{message} [{percentage:.1f}%] ({current}/{total})")

    def clear(self) -> None:
        """Clear the console"""
        with self._lock:
            self.console.clear()

    def input(self, prompt: str) -> str:
        """Thread-safe input with proper formatting"""
        with self._lock:
            return self.console.input(f"[yellow]{prompt}[/yellow] ")


# First, let's create a mock order generator for testing
class MockOrderGenerator:
    """Generates realistic test orders"""

    def __init__(self, rate: int = 15):  # 15 orders per minute default
        self.rate = rate
        self.order_id = 0

    async def generate_orders(self) -> List[Dict]:
        """Generate a batch of orders"""
        num_orders = random.randint(1, self.rate)
        orders = []

        for _ in range(num_orders):
            self.order_id += 1
            orders.append(
                {
                    "id": f"ORDER-{self.order_id}",
                    "timestamp": datetime.now().isoformat(),
                    "amount": round(random.uniform(10, 100), 2),
                    "status": "pending",
                }
            )

        # Simulate network delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        return orders


# Test-specific OrderProcessor with error injection
class TestOrderProcessor:
    """Order processor with error injection capabilities"""

    def __init__(self):
        self.order_queue = asyncio.Queue()
        self.chat_windows: Dict[str, "TestChatSession"] = {}
        self.error_scenarios = defaultdict(int)
        self.metrics = defaultdict(list)

    async def process_with_errors(self, order: Dict) -> bool:
        """Process order with controlled error injection"""
        # Simulate random processing errors
        if random.random() < 0.1:  # 10% chance of error
            self.error_scenarios["processing_error"] += 1
            raise Exception(f"Processing error for order {order['id']}")

        # Simulate timeout
        if random.random() < 0.05:  # 5% chance of timeout
            self.error_scenarios["timeout"] += 1
            await asyncio.sleep(5)
            raise asyncio.TimeoutError()

        await asyncio.sleep(random.uniform(0.1, 0.5))
        return True


# ====================== DRIVER CREATION ======================


# ====================== SESSION MANAGEMENT ======================
def save_session_data(app_state: AppState) -> bool:
    """Save current session data to a pickle file."""
    try:
        # Validate data before saving
        if not all(isinstance(v, (str, type(None))) for v in app_state.__dict__.values()):
            ui.status("Invalid data type in session data", "error")
            logging.error("Invalid data type in session data")
            return False

        with open("session.pkl", "wb") as f:
            pickle.dump(app_state.__dict__, f)
        return True
    except (IOError, pickle.PicklingError) as e:
        ui.status(f"Failed to save session: {str(e)}", "error")
        logging.error(f"Failed to save session: {e}")
        return False
    except Exception as e:
        ui.status(f"Unexpected error saving session: {str(e)}", "error")
        logging.error(f"Unexpected error saving session: {e}")
        return False


def load_session_data(app_state: AppState) -> bool:
    """Load previous session data."""
    try:
        with open("session.pkl", "rb") as f:
            data = pickle.load(f)
            app_state.customer_name = data.get("customer_name")
            app_state.customer_email = data.get("customer_email")
            app_state.customer_phone = data.get("customer_phone")
            app_state.num_orders = data.get("num_orders")
            app_state.restaurant_name = data.get("restaurant_name")
            return all(
                [
                    app_state.customer_name,
                    app_state.customer_email,
                    app_state.customer_phone,
                    app_state.num_orders,
                    app_state.restaurant_name,
                ]
            )
    except FileNotFoundError:
        ui.status("No previous session data found.", "info")
        logging.info("No previous session data found.")
        return False
    except (IOError, pickle.UnpicklingError) as e:
        ui.status(f"Error loading session data: {str(e)}", "warning")
        logging.warning(f"Error loading session data: {e}")
        return False


# ====================== AUTO LOGIN ======================
def auto_login(driver: uc.Chrome) -> bool:
    """Attempt automatic login with saved cookies."""
    try:
        with open("cookies.pkl", "rb") as f:
            cookies = pickle.load(f)
        driver.get("https://www.doordash.com/")
        for cookie in cookies:
            if cookie.get("domain") == ".www.doordash.com":
                cookie["domain"] = ".doordash.com"
            driver.add_cookie(cookie)
        driver.get("https://www.doordash.com/home")
        WebDriverWait(driver, config.get("TIMEOUTS.FAST_RECOVERY_TIMEOUT")).until(
            lambda d: "doordash.com/home" in d.current_url
        )

        # Try to collect customer info
        if info := CustomerManager.collect_from_website(driver):
            app_state.cookie_customer = info
            ui.status(f"Cookie customer info collected: {info}", "success")
        else:
            ui.status("Automatic info collection failed", "warning")
            app_state.cookie_customer = CustomerManager.manual_entry("Cookie")

        return True
    except FileNotFoundError:
        ui.status("No saved cookies found for auto-login.", "info")
        return False
    except TimeoutException:
        ui.status("Auto-login timed out.", "warning")
        return False
    except Exception as e:
        ui.status(f"Auto-login failed: {str(e)}", "warning")
        return False


# ====================== USER INPUT ======================
def get_user_input(
    prompt_text: str,
    required: bool = True,
    allow_empty: bool = False,
    choices: Optional[List[str]] = None,
) -> str:
    """Get user input with rich styling."""

    if choices:
        return Prompt.ask(f"[yellow]{prompt_text}[/yellow]", choices=choices, console=ui.console)

    while True:
        try:
            value = Prompt.ask(f"[yellow]{prompt_text}[/yellow]", console=ui.console).strip()

            if allow_empty and value == "":
                return value

            if not required:
                return value

            if required and not value:
                ui.console.print("[bold red]This field is required[/bold red]")
                continue

            return value

        except KeyboardInterrupt:
            return ""


# Add caching for frequently accessed elements
from functools import lru_cache


@lru_cache(maxsize=100)
def get_element_safely(driver: uc.Chrome, by_method: str, selector: str, timeout: int = 10) -> Optional[WebElement]:
    """Safely get element with caching"""
    try:
        return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by_method, selector)))
    except TimeoutException:
        return None


# Add batch processing for orders
def process_orders_batch(orders: List[Order], batch_size: int = 30) -> None:
    """Process orders in efficient batches"""
    for i in range(0, len(orders), batch_size):
        batch = orders[i : i + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            executor.map(process_single_order, batch)


class CustomerManager:
    """Manages customer information collection and storage."""

    def __init__(self, app_state: AppState):
        self.app_state = app_state

    def collect_from_website(self, driver: uc.Chrome) -> Optional[CustomerInfo]:
        """Collect customer information from the website."""
        try:
            # Navigate to edit profile
            driver.get("https://www.doordash.com/consumer/edit_profile/")
            time.sleep(2)

            # Strategy 1: Data-testid attributes
            first_name = None
            last_name = None
            email = None

            # Try primary selectors
            first_name_elem = get_element_safely(driver, By.CSS_SELECTOR, '[data-testid="givenName_input"]')
            last_name_elem = get_element_safely(driver, By.CSS_SELECTOR, '[data-testid="familyName_input"]')
            email_elem = get_element_safely(driver, By.CSS_SELECTOR, 'input[type="email"]')

            # Fallback for name if needed
            if not (first_name_elem and last_name_elem):
                name_elem = get_element_safely(driver, By.CSS_SELECTOR, '[data-testid="ConsumerAccountDetailsName"]')
                if name_elem:
                    name_parts = name_elem.text.strip().split(maxsplit=1)
                    first_name = name_parts[0]
                    last_name = name_parts[1] if len(name_parts) > 1 else ""
            else:
                first_name = first_name_elem.get_attribute("value")
                last_name = last_name_elem.get_attribute("value")

            # Get email value
            if email_elem:
                email = email_elem.get_attribute("value")

            # Validate and store
            if first_name and email:
                self.app_state.customer_first_name = first_name
                self.app_state.customer_last_name = last_name or ""
                self.app_state.customer_email = email
                logging.info(f"Customer info collected: {first_name} {last_name} ({email})")
                return CustomerInfo(first_name, last_name, email)

            raise ValueError("Could not collect required customer information")

        except Exception as e:
            logging.error(f"Failed to get customer info: {e}")
            return None

    def manual_entry(self, label: str) -> CustomerInfo:
        """Collect customer information manually."""
        try:
            ui.status(f"{label} customer info not found. Please enter customer details:", "warning")
            first_name = get_user_input("First Name: ")
            last_name = get_user_input("Last Name: ")
            email = get_user_input("Email: ")
            return CustomerInfo(first_name, last_name, email)
        except Exception as e:
            logging.error(f"Manual info entry failed: {e}")
            return None


@dataclass
class CustomerInfo:
    """Represents customer information."""

    first_name: str
    last_name: str
    email: str


# ====================== ORDER CLASS ======================


@dataclass
class AppState:
    """Application state management class"""

    customer_name: str = ""
    customer_email: str = ""
    customer_phone: str = ""
    num_orders: str = ""
    restaurant_name: str = ""
    customer_first_name: str = ""
    customer_last_name: str = ""
    cookie_customer: Optional["CustomerInfo"] = None
    is_logged_in: bool = False


# Create global app_state instance
app_state = AppState()


# 1. Enhanced Error Recovery System
class ErrorRecovery:
    """Intelligent error recovery system with retry logic"""

    def __init__(self):
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_errors: Dict[str, float] = defaultdict(float)
        self.recovery_strategies: Dict[str, Callable] = {
            "StaleElementReferenceException": self.handle_stale_element,
            "TimeoutException": self.handle_timeout,
            "WebDriverException": self.handle_driver_error,
        }

    async def handle_error(self, error: Exception, context: str) -> bool:
        """Smart error handling with context awareness"""
        error_type = error.__class__.__name__
        current_time = time.time()

        # Reset error counts if last error was more than 5 minutes ago
        if current_time - self.last_errors.get(context, 0) > 300:
            self.error_counts[context] = 0

        self.error_counts[context] += 1
        self.last_errors[context] = current_time

        if self.error_counts[context] > 3:
            logging.error(f"Too many errors in context {context}: {error}")
            return False

        if error_type in self.recovery_strategies:
            return await self.recovery_strategies[error_type](error, context)

        return False

    async def handle_stale_element(self, error: Exception, context: str) -> bool:
        """Handle stale element references"""
        logging.warning(f"Stale element in {context}, refreshing page...")
        await asyncio.sleep(1)
        return True


# 2. Advanced Session Management
class SessionManager:
    """Manages browser sessions and cookies"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.cookie_jar = CookieJar()
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str) -> None:
        """Create new session with proper initialization"""
        async with self._lock:
            if session_id in self.sessions:
                await self.cleanup_session(session_id)

            self.sessions[session_id] = {
                "created_at": datetime.now(),
                "last_active": datetime.now(),
                "cookies": {},
                "headers": self._get_default_headers(),
                "state": SessionState(),
            }

    async def save_session(self, session_id: str) -> bool:
        """Save session data to disk"""
        try:
            session_data = self.sessions.get(session_id)
            if not session_data:
                return False

            file_path = f"sessions/{session_id}.json"
            os.makedirs("sessions", exist_ok=True)

            async with aiofiles.open(file_path, "w") as f:
                await f.write(
                    json.dumps(
                        {
                            "cookies": session_data["cookies"],
                            "state": session_data["state"].to_dict(),
                            "created_at": session_data["created_at"].isoformat(),
                            "last_active": session_data["last_active"].isoformat(),
                        },
                        indent=2,
                    )
                )

            return True
        except Exception as e:
            logging.error(f"Failed to save session {session_id}: {e}")
            return False


# 3. Advanced Order Processing System
class OrderProcessor:
    """Optimized order processing for high-frequency chat operations"""

    def __init__(self):
        self.order_queue = asyncio.Queue()
        self.chat_windows: Dict[str, ChatSession] = {}
        self.metrics = MetricsCollector()
        self.config = {
            "ORDER_BATCH_SIZE": 30,  # Grab orders in batches of 30
            "ORDER_POLL_INTERVAL": 20,  # Poll every 20 seconds
            "CHAT_WINDOW_LIMIT": 15,  # Max concurrent chat windows
            "CHAT_TIMEOUT": 300,  # 5 minutes max per chat
            "RETRY_DELAY": 2,  # 2 seconds between retries
        }

    async def start_processing(self):
        """Main processing loop with concurrent tasks"""
        async with asyncio.TaskGroup() as tg:
            # Order collection task (runs every 20 seconds)
            tg.create_task(self.order_collector())
            # Chat window manager (continuous)
            tg.create_task(self.chat_window_manager())
            # Metrics reporter (runs every 60 seconds)
            tg.create_task(self.metrics_reporter())

    async def order_collector(self):
        """Collect orders efficiently"""
        while True:
            try:
                start_time = time.time()

                # Collect orders in batch
                new_orders = await self.fetch_new_orders()
                collection_time = time.time() - start_time

                await self.metrics.record_metric("order_collection_time", collection_time)

                # Process collected orders
                for order in new_orders:
                    await self.order_queue.put(order)

                # Adaptive sleep based on order volume
                sleep_time = self._calculate_sleep_time(len(new_orders))
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logging.error(f"Order collection error: {e}")
                await asyncio.sleep(self.config["RETRY_DELAY"])

    async def chat_window_manager(self):
        """Manage chat windows efficiently"""
        while True:
            try:
                # Check current chat window count
                active_chats = len(self.chat_windows)

                if active_chats < self.config["CHAT_WINDOW_LIMIT"]:
                    # We can open more chat windows
                    available_slots = self.config["CHAT_WINDOW_LIMIT"] - active_chats

                    for _ in range(available_slots):
                        if self.order_queue.empty():
                            break

                        order = await self.order_queue.get()
                        await self.open_chat_window(order)

                # Clean up completed/timed out chats
                await self.cleanup_chat_windows()

                await asyncio.sleep(1)  # Short sleep to prevent CPU overuse

            except Exception as e:
                logging.error(f"Chat window manager error: {e}")
                await asyncio.sleep(self.config["RETRY_DELAY"])

    async def open_chat_window(self, order: Dict):
        """Open chat window with timing metrics"""
        start_time = time.time()
        try:
            chat_session = ChatSession(order["id"])
            await chat_session.initialize()
            self.chat_windows[order["id"]] = chat_session

            window_open_time = time.time() - start_time
            await self.metrics.record_metric("chat_window_open_time", window_open_time)

        except Exception as e:
            logging.error(f"Failed to open chat for order {order['id']}: {e}")
            # Put order back in queue for retry
            await self.order_queue.put(order)

    async def cleanup_chat_windows(self):
        """Clean up completed or timed out chat windows"""
        current_time = time.time()
        to_remove = []

        for order_id, chat_session in self.chat_windows.items():
            if current_time - chat_session.start_time > self.config["CHAT_TIMEOUT"] or chat_session.is_completed:
                to_remove.append(order_id)
                await chat_session.cleanup()

        for order_id in to_remove:
            del self.chat_windows[order_id]

    def _calculate_sleep_time(self, orders_count: int) -> float:
        """Calculate adaptive sleep time based on order volume"""
        if orders_count == 0:
            return self.config["ORDER_POLL_INTERVAL"]
        elif orders_count < 10:
            return 15  # More frequent checks for low volume
        elif orders_count < 20:
            return 10  # Even more frequent for medium volume
        else:
            return 5  # Very frequent for high volume

    async def metrics_reporter(self):
        """Report performance metrics"""
        while True:
            metrics = {
                "active_chats": len(self.chat_windows),
                "queue_size": self.order_queue.qsize(),
                "avg_collection_time": await self.metrics.get_average("order_collection_time"),
                "avg_window_open_time": await self.metrics.get_average("chat_window_open_time"),
            }

            logging.info(f"Performance Metrics: {metrics}")
            await asyncio.sleep(60)


# 4. Enhanced UI System with Rich Integration


class EnhancedUI:
    """Advanced UI system with rich formatting"""

    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        self.live = Live(console=self.console, refresh_per_second=4)

    async def update_status(self, status: Dict) -> None:
        """Update status display with rich formatting"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        for key, value in status.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        self.live.update(Panel(table, title="[bold]Status Dashboard[/bold]", border_style="bright_blue"))

    def show_error(self, error: Exception, context: str = "") -> None:
        """Display error with rich formatting"""
        error_panel = Panel(
            Group(Text(str(error), style="red"), Text(f"Context: {context}", style="dim")),
            title="[red]Error Occurred[/red]",
            border_style="red",
        )
        self.console.print(error_panel)


class EnhancedUI(UI):
    def __init__(self):
        super().__init__()
        self.progress_bars = {}

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

    def order_summary(self, orders: List[Order]):
        """Display a pretty summary of orders"""
        with self._lock:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Order ID")
            table.add_column("Amount")
            table.add_column("Status")

            for order in orders:
                status_color = "green" if not order.is_canceled else "red"
                table.add_row(
                    order.short_id, f"${order.amount:.2f}", f"[{status_color}]{order.status}[/{status_color}]"
                )

            self.console.print(Panel(table, title="Order Summary"))

    def batch_progress(self, current: int, total: int, message: str = ""):
        """Show batch processing progress with a fancy progress bar"""
        with self._lock:
            percentage = (current / total) * 100
            bar_width = 40
            filled = int(bar_width * current / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            self.console.print(f"{message} [{percentage:>3.0f}%] |{bar}| ({current}/{total})", end="\r")


def process_orders_mode1(driver: uc.Chrome) -> None:
    ui = EnhancedUI()
    original_handle = driver.current_window_handle
    processed_orders = set()

    ui.status("🚀 Starting Mode 1 processing...", "info")

    while not exit_flag.is_set():
        try:
            # Collect orders with fancy progress
            with ui.status_spinner("Collecting orders..."):
                orders = get_orders(driver, Config.MAX_ORDERS)

            # Show order summary
            ui.order_summary(orders)

            # Process batch with enhanced progress tracking
            valid_orders = [o for o in orders if o.id not in processed_orders and o.amount > 5 and not o.is_canceled]

            if valid_orders:
                batch = valid_orders[: Config.BATCH_SIZE]
                for idx, order in enumerate(batch, 1):
                    ui.batch_progress(idx, len(batch), f"Processing order {order.short_id}")
                    # Your existing order processing logic here

            ui.status(f"✨ Completed batch of {len(batch)} orders", "success")

        except WebDriverException as e:
            ui.status(f"🚨 Browser error: {str(e)}", "error")


# 5. Metrics Collection System
class MetricsCollector:
    """Collect and analyze performance metrics"""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_metric(self, name: str, value: float) -> None:
        """Record a metric value"""
        async with self._lock:
            self.metrics[name].append(value)

            # Keep only last 1000 values
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistical analysis of a metric"""
        values = self.metrics.get(metric_name, [])
        if not values:
            return {}

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }


@dataclass
class Order:
    """Represents an order."""

    receipt_url: str
    id: str
    short_id: str
    url: str
    tab_handle: Optional[str] = None
    wait_time: Optional[int] = None
    amount: float = 0.0
    is_canceled: bool = False

    def __init__(self, order_element: uc.WebElement):
        """Initialize order from receipt link element."""
        try:
            self.receipt_url = order_element.find_elements(By.TAG_NAME, "a")[-1].get_attribute("href")
            if not self.receipt_url:
                raise ValueError("No receipt URL found")
            self.id = self.receipt_url.split("/orders/")[-1].replace("/receipt/", "").split("?")[0]
            self.short_id = self.id[-6:].upper()  # Store short ID version
            self.url = f"https://doordash.com/orders/{self.id}/help/"
            self.tab_handle = None
            self.wait_time = None
            self.amount = float(
                get_element_from_text(order_element, "span", " item", exact=False).text.split(" • ")[1].replace("$", "")
            )
            self.is_canceled = (
                get_element_from_text(order_element, "span", "Order Cancelled", exact=False) is not None
                or get_element_from_text(order_element, "span", "Refund", exact=False) is not None
            )
        except Exception as e:
            ui.status(f"Error initializing order: {str(e)}", "error")
            logging.error(f"Error initializing order: {e}")
            raise

    def open_support_chat(self, driver: uc.Chrome) -> bool:
        """Open support chat with improved retry logic and element waiting."""
        max_retries = 4
        wait = WebDriverWait(driver, config.get("TIMEOUTS.ELEMENT_TIMEOUT"))

        for retry in range(max_retries):
            try:
                driver.get(self.url)
                if countdown_timer(2, "Loading support page"):
                    return False

                something_else = wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//button[@aria-label="It\'s something else"]'))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", something_else)
                if countdown_timer(0.3, "Ensuring element is clickable"):
                    return False
                something_else.click()
                ui.status("Clicked 'It's something else'", "success", transient=True)
                logging.info("Clicked 'It's something else'")
                if countdown_timer(0.1, "Waiting after click"):
                    return False
                break
            except (WebDriverException, TimeoutException) as e:
                if retry == max_retries - 1:
                    ui.status(
                        f"Failed to find 'It's something else' button: {str(e)}",
                        "error",
                        transient=True,
                    )
                    logging.error(f"Failed to find 'It's something else' button: {e}")
                countdown_timer(0.5, "Refreshing page")
                driver.refresh()
                continue

        for retry in range(max_retries):
            try:
                contact = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Contact support')]")))
                driver.execute_script("arguments[0].scrollIntoView(true);", contact)
                if countdown_timer(0.5, "Ensuring element is clickable"):
                    return False
                contact.click()
                ui.status("Clicked 'Contact support'", "success", transient=True)
                logging.info("Clicked 'Contact support'")
                if countdown_timer(0.5, "Waiting for chat window"):
                    return False
                return True
            except (WebDriverException, TimeoutException) as e:
                if retry == max_retries - 1:
                    ui.status(f"Failed to find 'Contact support' button: {str(e)}", "error")
                    logging.error(f"Failed to find 'Contact support' button: {e}")
                countdown_timer(2, "Retrying support button")
        return False

    def send_message_to_support(self, message: str, driver: uc.Chrome) -> bool:
        """Send message with improved handling."""
        max_retries = 3
        wait = WebDriverWait(driver, config.get("TIMEOUTS.ELEMENT_TIMEOUT"))

        for attempt in range(max_retries):
            try:
                text_area = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "textarea, [role='textbox'], .chat-input"))
                )

                # Focus and clear the text area
                try:
                    text_area.clear()
                    text_area.click()
                except Exception:
                    ActionChains(driver).move_to_element(text_area).click().perform()

                time.sleep(0.1)  # Wait for focus

                # Process message line by line
                lines = message.split("\n")
                for i, line in enumerate(lines):
                    if not line.strip():  # Add empty line
                        ActionChains(driver).key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
                        continue

                    if i > 0 and lines[i - 1].startswith("PHONE:"):
                        ActionChains(driver).key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()

                    # New line for non-first lines
                    if i > 0:
                        ActionChains(driver).key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()

                    text_area.send_keys(line)
                    time.sleep(0.1)

                time.sleep(0.1)  # Wait before final send
                text_area.send_keys(Keys.RETURN)
                countdown_timer(0.1, "Waiting after sending message")
                return True

            except (WebDriverException, TimeoutException) as e:
                ui.status(
                    f"Send message attempt {attempt + 1}/{max_retries} failed: {str(e)}",
                    "error",
                )
                logging.error(f"Send message attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    countdown_timer(3, "Retrying send message")
                    driver.refresh()
                    countdown_timer(3, "Waiting after refresh")
                else:
                    return False
        return False

    def check_wait_time(self, driver: uc.Chrome) -> Optional[int]:
        """Check for wait time message in chat window."""
        try:
            messages = driver.find_elements(By.CSS_SELECTOR, ".chat-message, .message-text")
            for msg in messages:
                if wait_time := extract_wait_time(msg.text):
                    self.wait_time = wait_time
                    ui.status(
                        f"Wait time for Order #{self.short_id}: {self.wait_time}m",
                        "info",
                    )
                    logging.info(f"Wait time for Order #{self.short_id}: {self.wait_time}m")
                    return wait_time
        except Exception:
            pass
        return None


# ====================== STATUS HANDLER ======================
class StatusHandler:
    """Handles status messages during order processing."""

    def __init__(self):
        self.console = Console()

    def error_status(self, message: str):
        ui.status(message, "error")
        logging.error(message)

    def wait_status(self, message: str):
        ui.status(message, "info")
        logging.info(message)

    def click_status(self, message: str):
        ui.status(message, "success", transient=True)
        logging.info(message)


# ====================== ORDER PROCESSING ======================
def process_orders_mode2(orders: List[Order], driver: uc.Chrome) -> None:
    """Process orders with improved memory management and error handling"""
    original_handle = driver.current_window_handle
    active_chats: Dict[str, ChatStatus] = {}
    orders_queue = deque(orders)
    batch_size = min(30, config.get("BROWSER_SETTINGS.MAX_OPEN_TABS", 30))

    try:
        while orders_queue and not exit_flag.is_set():
            batch_orders = [orders_queue.popleft() for _ in range(min(batch_size, len(orders_queue)))]

            for order in batch_orders:
                try:
                    if process_single_order(driver, order, active_chats):
                        stats["sent"] += 2
                    else:
                        stats["errors"] += 1
                except Exception as e:
                    logging.error(f"Error processing order {order.short_id}: {e}")
                    stats["errors"] += 1
                finally:
                    driver.switch_to.window(original_handle)

            if len(active_chats) >= batch_size:
                monitor_chats(driver, active_chats)

    except Exception as e:
        logging.error(f"Batch processing error: {e}")
    finally:
        if active_chats:
            monitor_chats(driver, active_chats)


def process_single_order(driver: uc.Chrome, order: Order, active_chats: Dict[str, ChatStatus]) -> bool:
    """Process a single order with proper error handling"""
    try:
        driver.switch_to.new_window("tab")
        new_handle = driver.window_handles[-1]
        order.tab_handle = new_handle

        if not order.open_support_chat(driver):
            return False

        message = get_remove_tip_message(app_state)
        if not order.send_message_to_support(message, driver):
            return False

        active_chats[order.id] = ChatStatus(order_id=order.short_id, tab_handle=new_handle, message_sent=True)
        return True

    except Exception as e:
        logging.error(f"Order processing error: {e}")
        try:
            driver.close()
        except:
            pass
        return False


def get_element_safely(driver: uc.Chrome, by_method: str, selector: str, timeout: int = 10) -> Optional[WebElement]:
    """Safely get element with timeout."""
    try:
        return WebDriverWait(driver, timeout).until(EC.presence_of_element_located((by_method, selector)))
    except TimeoutException:
        return None


def get_customer_info(driver: uc.Chrome, app_state: AppState) -> bool:
    """Extract customer information using multiple strategies."""
    try:
        # Navigate to edit profile
        driver.get("https://www.doordash.com/consumer/edit_profile/")
        time.sleep(2)

        # Strategy 1: Data-testid attributes
        first_name = None
        last_name = None
        email = None

        # Try primary selectors
        first_name_elem = get_element_safely(driver, By.CSS_SELECTOR, '[data-testid="givenName_input"]')
        last_name_elem = get_element_safely(driver, By.CSS_SELECTOR, '[data-testid="familyName_input"]')
        email_elem = get_element_safely(driver, By.CSS_SELECTOR, 'input[type="email"]')

        # Fallback for name if needed
        if not (first_name_elem and last_name_elem):
            name_elem = get_element_safely(driver, By.CSS_SELECTOR, '[data-testid="ConsumerAccountDetailsName"]')
            if name_elem:
                name_parts = name_elem.text.strip().split(maxsplit=1)
                first_name = name_parts[0]
                last_name = name_parts[1] if len(name_parts) > 1 else ""
        else:
            first_name = first_name_elem.get_attribute("value")
            last_name = last_name_elem.get_attribute("value")

        # Get email value
        if email_elem:
            email = email_elem.get_attribute("value")

        # Validate and store
        if first_name and email:
            app_state.customer_first_name = first_name
            app_state.customer_last_name = last_name or ""
            app_state.customer_email = email
            logging.info(f"Customer info collected: {first_name} {last_name} ({email})")
            return True

        raise ValueError("Could not collect required customer information")

    except Exception as e:
        logging.error(f"Failed to get customer info: {e}")
        return False


def manual_customer_info_entry(app_state: AppState) -> bool:
    """Collect customer info manually."""
    try:
        ui.status("Auto-collection failed. Please enter customer details:", "warning")
        app_state.customer_first_name = get_user_input("First Name: ")
        app_state.customer_last_name = get_user_input("Last Name: ")
        app_state.customer_email = get_user_input("Email: ")
        return True
    except Exception as e:
        logging.error(f"Manual info entry failed: {e}")
        return False


def ensure_customer_info(driver: uc.Chrome, app_state: AppState) -> bool:
    """Ensure customer info is available through auto or manual collection."""
    # Try auto collection first
    if get_customer_info(driver, app_state):
        return True

    # Fallback to manual entry
    return manual_customer_info_entry(app_state)


""" #// BUILD PLAN
#// 1, CORE COMPONENTS
#// - Config class
#//- Order collection/filtering
#// - Chat monitoring with timeouts
#// - Status display system
Processing Flow


a. Initialize:
   - Config settings
   - WebDriver state
   - Order tracking

b. Main Loop:
   - Get orders (up to 200)
   - Filter valid orders (min 5)
   - Process in batches (30)
   - Monitor chats
   - Track progress

c. Chat Monitoring:
   - Wait 30s
   - Refresh all tabs
   - Wait 30s more
   - Close tabs
   - Return to orders page """


class Config:
    """System configuration."""

    BATCH_SIZE = 30
    WAIT_TIME = 30
    MAX_ORDERS = 200
    MIN_ORDERS = 5
    CHECK_INTERVAL = 60


def process_orders_mode1(driver: uc.Chrome) -> None:
    """Main processing loop for Mode 1."""
    original_handle = driver.current_window_handle
    processed_orders = set()

    with Live() as live:
        while not exit_flag.is_set():
            try:
                # Get orders
                with ui.progress_bar() as progress:
                    task = progress.add_task("Loading orders...", total=None)
                    orders = get_orders(driver, Config.MAX_ORDERS)
                    progress.update(task, completed=True)

                # Filter valid orders
                valid_orders = [
                    o for o in orders if o.id not in processed_orders and o.amount > 5 and not o.is_canceled
                ]

                if len(valid_orders) < Config.MIN_ORDERS:
                    ui.status(f"Waiting for orders ({len(valid_orders)})", "info")
                    time.sleep(10)
                    continue

                # Process batch
                batch = valid_orders[: Config.BATCH_SIZE]
                active_chats = {}

                # Process orders with progress
                with ui.progress_bar() as progress:
                    task = progress.add_task("Processing orders...", total=len(batch))

                    for order in batch:
                        try:
                            driver.switch_to.new_window("tab")
                            if order.open_support_chat(driver):
                                message = get_remove_tip_message1(app_state)
                                if order.send_message_to_support(message, driver):
                                    active_chats[order.id] = ChatStatus(
                                        order_id=order.id, tab_handle=driver.current_window_handle
                                    )
                                    processed_orders.add(order.id)
                            progress.update(task, advance=1)
                        except Exception as e:
                            logging.error(f"Error processing {order.id}: {e}")
                            driver.close()
                        finally:
                            driver.switch_to.window(original_handle)

                # Monitor chats if any active
                if active_chats:
                    monitor_chats(driver, active_chats)

            except WebDriverException as e:
                ui.status(f"Browser error: {str(e)}", "error")
                if not handle_browser_disconnect(driver):
                    break

            # Brief pause before next batch
            time.sleep(5)

    return processed_orders


def monitor_chats(driver: uc.Chrome, active_chats: Dict[str, ChatStatus]) -> None:
    """Monitor active chat windows with progress tracking."""
    if not active_chats:
        return

    original_handle = driver.current_window_handle
    valid_handles = set(driver.window_handles)
    chats_to_remove = set()
    WAIT_TIME = 30

    try:
        # First wait period with progress
        with ui.progress_bar() as progress:
            task = progress.add_task("Initial wait...", total=WAIT_TIME)
            for i in range(WAIT_TIME):
                time.sleep(1)
                progress.update(task, advance=1)

        # Refresh all chat tabs
        ui.status(f"Refreshing {len(active_chats)} chats...", "info")
        for order_id, chat in active_chats.items():
            try:
                if chat.tab_handle not in valid_handles:
                    chats_to_remove.add(order_id)
                    continue

                driver.switch_to.window(chat.tab_handle)
                driver.refresh()
                time.sleep(0.5)

            except WebDriverException:
                chats_to_remove.add(order_id)
                continue

        # Second wait period
        with ui.progress_bar() as progress:
            task = progress.add_task("Final wait...", total=WAIT_TIME)
            for i in range(WAIT_TIME):
                time.sleep(1)
                progress.update(task, advance=1)

        # Close all tabs
        ui.status(f"Closing {len(active_chats)} chats...", "info")
        for order_id in chats_to_remove:
            try:
                chat = active_chats[order_id]
                driver.switch_to.window(chat.tab_handle)
                driver.close()
                del active_chats[order_id]
            except:
                continue

    except Exception as e:
        logging.error(f"Chat monitoring error: {e}")
    finally:
        try:
            driver.switch_to.window(original_handle)
        except WebDriverException:
            if driver.window_handles:
                driver.switch_to.window(driver.window_handles[0])


# ui is used but not properly initialized
class ConsoleUI:
    def __init__(self):
        self.console = Console()

    def status(self, message: str, status_type: str = "info", transient: bool = False):
        # Implementation needed
        pass


ui = ConsoleUI()


# ====================== ORDER COLLECTION ======================
class OrderCollector:
    def __init__(self, driver: WebDriver, config: AppConfig):
        self.driver = driver
        self.config = config
        self.processed_orders = set()
        self.last_order_id = None


# config is used but not properly initialized
config = {
    "TIMEOUTS": {
        "PAGE_LOAD_TIMEOUT": 30,
        "ELEMENT_TIMEOUT": 10,
        "FAST_RECOVERY_TIMEOUT": 5,
        "CHAT_CHECK_INTERVAL": 60,
    },
    "MAX_SCROLL_ATTEMPTS": 50,
    "MAX_BATCH_SIZE": 30,
}


class OrderProcessor:
    def __init__(self, driver: WebDriver, config: AppConfig):
        self.driver = driver
        self.config = config
        self.collector = OrderCollector(driver, config)
        self.batch_size = config.get("MAX_BATCH_SIZE", 30)
        self.app_state = app_state  # Add reference to global app_state

    async def process_orders(self):
        """Process orders asynchronously."""
        try:
            # Display login menu and handle login
            menu_ui = MenuUI(self.app_state)  # Use instance app_state
            if not menu_ui.handle_login(self.driver):
                return

            # Get mode selection
            selected_mode = menu_ui.display_mode_menu()
            if selected_mode == "5":
                return

            if selected_mode == "1":
                ui.status("Starting Mode 1 - REMOVAL WHILE RUNS!", "info")
                process_orders_mode1(self.driver)
            elif selected_mode == "2":
                ui.status("Starting Mode 2 - REMOVAL OF DEACTIVATED", "info")
                orders = get_orders(self.driver)
                if orders:
                    display_order_summary(orders)
                    process_orders_mode2(orders, self.driver)

        except Exception as e:
            ui.status(f"Error processing orders: {str(e)}", "error")
            logging.error(f"Error processing orders: {e}")
            raise


def get_orders(driver: uc.Chrome, max_orders: int = None) -> List[Order]:
    ui = EnhancedUI()

    if max_orders is None:
        max_orders = config.get("MAX_BATCH_SIZE")

    ui.status("🔍 Starting order collection...", "info")
    driver.get(url="https://www.doordash.com/orders")

    # Show spinner while page loads
    for _ in ui.show_spinner("Loading orders page..."):
        time.sleep(2)  # Your existing wait time
        break

    orders = []
    seen_orders = set()
    scroll_attempts = 0
    last_height = 0

    while len(orders) < max_orders and scroll_attempts < config.get("MAX_SCROLL_ATTEMPTS", 50):
        ui.status(f"Found {len(orders)} orders so far...", "info", transient=True)
        # Your existing order collection code...

    ui.status(f"✨ Collected {len(orders)} orders successfully!", "success")
    return orders
    if max_orders is None:
        max_orders = config.get("MAX_BATCH_SIZE")

    orders = []
    ui.status("Starting order collection...", "debug")
    logging.debug("Starting order collection...")

    driver.get(url="https://www.doordash.com/orders")
    if countdown_timer(1, "Waiting for page load"):
        logging.debug("Page load timed out.")
        return []

    # Load all orders by clicking "Load More" until it's no longer available
    while True:
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.5)
            load_more = driver.find_element(By.XPATH, "//span[text()='Load More']")
            load_more.click()
            time.sleep(0.5)  # allow more time to load
        except Exception:
            break

    # Get the orders container element
    try:
        completed_span = driver.find_element(By.XPATH, "//span[text()='Completed']")
        orders_container_element = completed_span.find_element(By.XPATH, "..")
        orders_container_element = orders_container_element.find_element(By.XPATH, "./div[last()]")
        logging.debug(f"Orders container element found: {orders_container_element.text}")
    except Exception as e:
        logging.error(f"Could not find orders container element: {e}")
        return []

    # Loop through orders with infinite scroll handling
    scroll_attempts = 0
    last_height = 0
    orders = []
    seen_orders = set()

    while len(orders) < max_orders and scroll_attempts < config.get("MAX_SCROLL_ATTEMPTS", 50):
        # Fast scroll to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)  # Minimal wait for DOM updates

        # Get visible order elements
        order_elements = orders_container_element.find_elements(By.XPATH, "./div")

        for element in order_elements:
            try:
                # Skip if already processed
                order_id = element.get_attribute("data-order-id")
                if order_id in seen_orders:
                    continue

                # Create order object and validate
                order = Order(element)
                seen_orders.add(order_id)

                if order.amount > 5 and not order.is_canceled:
                    orders.append(order)
                    logging.debug(f"Added order: id={order.id}, amount={order.amount}")

                if len(orders) >= max_orders:
                    return orders

            except Exception as e:
                ui.status(f"Error creating order: {str(e)}", "error")
                logging.error(f"Error creating order: {e}", exc_info=True)
                continue

        # Check if we've reached bottom
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                load_more = driver.find_element(By.XPATH, "//span[text()='Load More']")
                load_more.click()
                time.sleep(0.5)
            except:
                break

        last_height = new_height
        scroll_attempts += 1

    return orders


def _is_processed(self, order_id: str) -> bool:
    return order_id in self.processed_orders


def mark_processed(self, order_id: str) -> None:
    self.processed_orders.add(order_id)

    async def process_orders(self):
        while True:
            # Get new batch of unprocessed orders
            orders = self.collector.get_orders(self.batch_size)
            if not orders:

                break

            # Process batch concurrently
            tasks = []
            for order in orders:
                tasks.append(self.process_single_order(order))

            results = await asyncio.gather(*tasks)

            # Mark successful orders
            for order, success in zip(orders, results):
                if success:
                    self.collector.mark_processed(order.id)

            # Check exit conditions
            if len(self.collector.processed_orders) >= self.config.get("MAX_ORDERS", 200):
                break

    async def process_single_order(self, order: Order) -> bool:
        try:
            await self.open_chat(order)
            await self.send_message(order)
            return True
        except Exception as e:
            logging.error(f"Failed to process order {order.id}: {e}")
            return False


# ====================== CHAT MONITORING ======================
def monitor_chats(driver: uc.Chrome, active_chats: Dict[str, ChatStatus]) -> None:
    """Monitor active chats with batch operations."""
    original_handle = driver.current_window_handle
    WAIT_TIME = 30  # seconds between operations

    try:
        # First wait period
        with ui.progress_bar() as progress:
            task = progress.add_task("Initial wait period...", total=WAIT_TIME)
            for i in range(WAIT_TIME):
                time.sleep(1)
                progress.update(task, advance=1)

        # Refresh all tabs
        ui.status(f"Refreshing {len(active_chats)} tabs...", "info")
        for order_id, chat in active_chats.items():
            try:
                driver.switch_to.window(chat.tab_handle)
                driver.refresh()
            except WebDriverException:
                continue

        # Second wait period
        with ui.progress_bar() as progress:
            task = progress.add_task("Final wait period...", total=WAIT_TIME)
            for i in range(WAIT_TIME):
                time.sleep(1)
                progress.update(task, advance=1)

        # Close all tabs
        ui.status(f"Closing {len(active_chats)} tabs...", "info")
        for order_id, chat in list(active_chats.items()):
            try:
                driver.switch_to.window(chat.tab_handle)
                driver.close()
                del active_chats[order_id]
            except WebDriverException:
                continue

    except Exception as e:
        logging.error(f"Error in chat monitoring: {e}")
    finally:
        # Return to main window
        try:
            driver.switch_to.window(original_handle)
        except WebDriverException:
            if driver.window_handles:
                driver.switch_to.window(driver.window_handles[0])

    # Wait with live countdown
    if len(active_chats) > 0:
        start_time = time.time()
        monitoring_interval = config.get("TIMEOUTS.CHAT_CHECK_INTERVAL", 60)  # Default 60 seconds if not configured
        with ui.console.status("[bold white]Monitoring chats...", spinner="dots") as status:
            while (time.time() - start_time) < monitoring_interval:
                remaining = monitoring_interval - (time.time() - start_time)
                status.update(f"[bold white]Monitoring {len(active_chats)} active chats... {remaining:.1f}s remaining")
                if check_keyboard_input():
                    break
                time.sleep(0.1)


# ====================== ORDER PLACEHOLDER ======================
class OrderPlaceholder(Order):
    """Placeholder Order class for checking wait time without a real element."""

    def __init__(self, order_id: str):
        self.receipt_url = ""
        self.id = order_id
        self.short_id = order_id[-6:].upper()
        self.url = ""
        self.tab_handle = None
        self.wait_time = None
        self.amount = 0.0
        self.is_canceled = False

    def check_wait_time(self, driver: uc.Chrome) -> Optional[int]:
        """Check wait time without needing a WebElement."""
        return extract_wait_time(driver.page_source)


# ====================== AGENT MESSAGE FUNCTIONS ======================
def get_remove_tip_message(app_state: AppState) -> str:
    """Generate the tip removal message for support."""
    try:
        num = int(app_state.num_orders) if app_state.num_orders else ""
        message = (
            f"PLS CHECK ACCOUNT WITH THESE DETAILS:\n"
            f"\n"
            f"EMAIL: {app_state.customer_email}\n"
            f"NAME: {app_state.customer_name}\n"
            f"PHONE: {app_state.customer_phone}\n"
            f"\n"
            f"REMOVE ALL TIPS ON ALL ORDERS ON THE ENTIRE ACCOUNT,\n"
            f"SCROLL AND CHECK ALL {num} {app_state.restaurant_name} ORDERS!!!\n"
            f"CHANGE ALL TIPS ON ALL ORDERS TO $0\n"
            f"\n"
            f" <3 UwU"
        )
        return message
    except Exception as e:
        ui.status(f"Error formatting message: {str(e)}", "error")
        logging.error(f"Error formatting message: {e}")
        return "Error generating message"


def get_remove_tip_message1(app_state: AppState) -> str:
    """Generate removal message with customer info."""
    options = [
        "Please remove the dasher tip to $0",
        "Hey, pls remove the tip and adjust it to $0",
        "Hi, i want you to remove whole dasher tip and make it $0",
        "Hey, remove full dasher tip and make it $0 pls. Application is glitching and it charged my card twice for the tip idk what is happening",
        "hey remove dasher's tip and adjust to $0",
    ]
    message = random.choice(options)

    return f"{app_state.customer_name}\n\n{app_state.customer_email}\n\n{message}"


def get_agent_message(agent_name: str, app_state: AppState) -> str:
    """Generate the agent-specific message."""
    num_orders = app_state.num_orders or "N/A"
    return (
        f"{agent_name}!!!\n\n"
        f"FIND THIS ACCOUNT:\n\n"
        f"EMAIL: {app_state.customer_email}\n"
        f"NAME: {app_state.customer_name}\n"
        f"PHONE: {app_state.customer_phone}\n\n"
        f"They are unable to log in :(\n\n"
        f"REMOVE ALL {num_orders} TIPS ON {app_state.customer_name}'s account.\n\n"
        f"HAVE YOU DONE THAT, {agent_name}??? Thx!!!\n\n"
        f"<3"
    )


def send_agent_message(driver: uc.Chrome, text_area: WebDriver, message: str) -> bool:
    """Send message to agent with proper line breaks."""
    try:
        # Focus the text area
        try:
            text_area.click()
            text_area.clear()
        except Exception:
            ActionChains(driver).move_to_element(text_area).click().perform()

        # Use JavaScript to set the value directly
        message_js = message.replace("\n", "\\n").replace('"', '\\"')
        driver.execute_script(f'arguments[0].value = "{message_js}";', text_area)

        # Trigger change event to ensure value is recognized
        driver.execute_script('arguments[0].dispatchEvent(new Event("change"));', text_area)

        # Send the message
        text_area.send_keys(Keys.RETURN)
        return True

    except Exception as e:
        ui.status(f"Failed to send agent message: {str(e)}", "error")
        logging.error(f"Failed to send agent message: {e}")
        return False


# ====================== MENU HANDLING ======================
class MenuUI:
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.console = Console()
        self.refresh_rate = 0.1  # Reduced refresh rate
        self.last_refresh = 0
        self.current_selection = 0
        self.mode_selection = None

    def display_menu(self, options: List[tuple], title: str, subtitle: str = None) -> str:
        """Unified menu display method with improved handling"""
        current_selection = 0

        while True:
            current_time = time.time()
            if current_time - self.last_refresh >= self.refresh_rate:
                self.last_refresh = current_time

                # Build menu content
                menu_content = []
                for idx, (key, text) in enumerate(options):
                    prefix = ">" if idx == current_selection else " "
                    style = "bold white on red" if idx == current_selection else "white"
                    menu_content.append(f"[{style}]{prefix} {key}: {text}[/]")

                # Create menu panel
                menu_panel = Panel(
                    "\n".join(menu_content),
                    title=f"[bold white]{title}[/]",
                    subtitle=subtitle or "[dim]Use arrows or number keys[/]",
                    box=DOUBLE,
                    expand=False,
                    width=60,
                    border_style="#D91400",
                    padding=(1, 2),
                )

                # Clear and display
                self.console.clear()
                self.console.print("\n" * 2)  # Add some spacing
                self.console.print(menu_panel)

            # Handle input
            if key := check_keyboard_input():
                if key == b"\xe0":  # Arrow keys
                    arrow = msvcrt.getch()
                    if arrow == b"H":  # Up
                        current_selection = (current_selection - 1) % len(options)
                    elif arrow == b"P":  # Down
                        current_selection = (current_selection + 1) % len(options)
                elif key in [b"1", b"2", b"5"]:  # Direct number selection
                    return key.decode()
                elif key == b"\r":  # Enter
                    return options[current_selection][0]

            time.sleep(0.05)  # Reduce CPU usage

    def display_login_menu(self) -> str:
        """Display login menu with status"""
        options = [("1", "Mode 1 - REMOVAL WHILE RUNS!"), ("2", "Mode 2 - REMOVAL OF DEACTIVATED"), ("5", "Exit")]

        # Get status info
        session_exists = os.path.exists("session.pkl")
        cookies_exist = os.path.exists("cookies.pkl")

        status_text = (
            f"Session Info: {'[green]Active[/]' if session_exists else '[red]Missing[/]'}\n"
            f"Cookies: {'[green]Active[/]' if cookies_exist else '[red]Missing[/]'}"
        )

        return self.display_menu(options, "LOGIN OPTIONS", f"[white]{status_text}[/]")

    def display_mode_menu(self) -> str:
        """Display mode selection menu"""
        options = [("1", "Manual Login!"), ("2", "Login with Cookies"), ("5", "Exit")]

        return self.display_menu(options, "MODE SELECTION", "[white]Select login method[/]")

    def handle_login(self, driver: uc.Chrome) -> bool:
        """Handle login process with improved error handling"""
        try:
            # Get initial mode selection
            choice = self.display_login_menu()
            if choice == "5":
                return False

            self.mode_selection = choice

            # Get login mode
            login_mode = self.display_mode_menu()
            if login_mode == "5":
                return False

            # Handle login based on selection
            if login_mode == "1":
                success = self.handle_manual_login(driver)
            else:
                success = self.handle_cookie_login(driver)

            # Verify login if successful
            if success:
                return self._verify_login(driver)

            return False

        except Exception as e:
            logging.error(f"Login error: {e}")
            return False

    def handle_manual_login(self, driver: uc.Chrome) -> bool:
        """Process manual login"""
        try:
            driver.get("https://www.doordash.com/consumer/login/")
            ui.status("Waiting for manual login...", "processing")

            while "/home" not in driver.current_url:
                if countdown_timer(0.5, "Waiting for login"):
                    return False

            # Save cookies after successful login
            cookies = driver.get_cookies()
            with open("cookies.pkl", "wb") as f:
                pickle.dump(cookies, f)
            ui.status("Login cookies saved successfully.", "success")

            # Collect customer info
            if info := CustomerManager.collect_from_website(driver):
                app_state.cookie_customer = info
                ui.status(f"Customer info collected: {info}", "success")
            else:
                ui.status("Manual info collection needed", "warning")
                app_state.cookie_customer = CustomerManager.manual_entry("New")

            return True

        except Exception as e:
            ui.status(f"Manual login failed: {str(e)}", "error")
            logging.error(f"Manual login failed: {e}")
            return False

    def handle_cookie_login(self, driver: uc.Chrome) -> bool:
        """Process cookie-based login"""
        if not auto_login(driver):
            ui.status("Auto login failed", "error")
            return False

        # Load previous session if available
        has_previous_session = load_session_data(self.app_state)

        # Don't show session info immediately - we'll show it later if Mode 2 is selected
        if has_previous_session:
            ui.status("Previous session details loaded", "success")
            return True

        # Only get new details if Mode 2 is selected (handled in main)
        self.app_state.customer_email = ""
        self.app_state.customer_name = ""
        self.app_state.customer_phone = ""
        self.app_state.num_orders = "10"
        self.app_state.restaurant_name = ""

        if has_previous_session:
            ui.status("Using previous session details", "success")
            show_session_info(self.app_state, ui)
            show_message_preview(self.app_state, ui)
            return True

    def _get_new_session_details(self):
        """Get new session details from user"""
        input_panel = self.create_panel("NEW CUSTOMER DETAILS", "session")
        self.console.print(input_panel)

        self.app_state.customer_email = get_user_input("Email", required=False) or ""
        self.app_state.customer_name = get_user_input("Full Name", required=False) or ""
        self.app_state.customer_phone = get_user_input("Phone Number", required=False) or ""
        self.app_state.num_orders = get_user_input("Number of orders to check", required=False) or "10"
        self.app_state.restaurant_name = get_user_input("Restaurant name", required=False) or ""

        if save_session_data(self.app_state):
            ui.status("Session details saved", "success")
            show_session_info(self.app_state, ui)
            show_message_preview(self.app_state, ui)
        else:
            ui.status("Failed to save session details", "error")


# ====================== HELPER FUNCTIONS ======================
def show_results(stats: dict) -> None:
    """Display final results with performance metrics."""
    perf_stats = perf_monitor.get_stats()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Messages Sent:", f"[green]{stats['sent']}")
    table.add_row("Errors:", f"[red]{stats['errors']}")
    table.add_row("Chats Checked:", f"[white]{stats['checked']}")
    table.add_row("Average Request Time:", f"[cyan]{perf_stats['avg_time']:.2f}s")
    table.add_row("Success Rate:", f"[green]{perf_stats['success_rate']:.1f}%")
    table.add_row("Total Requests:", f"[white]{perf_stats['total_requests']}")

    panel = Panel(
        table,
        title="[bold white]FINAL RESULTS",
        box=DOUBLE,
        expand=True,
        border_style="#D91400",
        padding=(0, 2),
    )
    ui.console.print(panel)


# Configure logging to only show critical in console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.CRITICAL)
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Remove existing handlers
logging.getLogger().handlers.clear()

# Add new handlers
logging.getLogger().addHandler(console_handler)
logging.getLogger().addHandler(file_handler)

# Add timestamps to logging
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)


def show_message_preview(app_state: AppState, ui: ConsoleUI) -> None:
    """Display the message template with current session details."""
    message = get_remove_tip_message(app_state)
    ui.console.print(
        Panel(
            Text(message, justify="left"),
            title=f"[bold white]{config.get('UI_SETTINGS.MESSAGE_PREVIEW_HEADER')}",
            box=DOUBLE,
            expand=True,
            border_style="#D91400",
            padding=(0, 2),
        )
    )


def show_session_info(app_state: AppState, ui: ConsoleUI) -> None:
    """Display current session information in a styled box."""
    details = {
        "Customer Name": app_state.customer_name,
        "Email Address": app_state.customer_email,
        "Phone Number": app_state.customer_phone,
        "Order Count": app_state.num_orders,
        "Restaurant": app_state.restaurant_name,
    }

    # Start countdown timer for auto-proceed
    start_time = time.time()
    duration = 30  # 30 seconds countdown

    with ui.console.status("[bold white]Starting...", spinner="dots") as status:
        while time.time() - start_time < duration:
            remaining = int(duration - (time.time() - start_time))
            status.update(f"[bold white]Proceeding in {remaining} seconds...")

            # Check for keyboard input inside the loop
            key = check_keyboard_input()
            if key:
                if key.lower() == b"n":
                    # Create input panel for new information
                    input_table = Table(
                        show_header=False,
                        box=None,
                        width=config.get("UI_SETTINGS.PANEL_WIDTH") - 4,
                        padding=(0, 2),
                    )

                    input_panel = Panel(
                        input_table,
                        title="[bold white]ENTER NEW DETAILS",
                        box=DOUBLE,
                        expand=True,
                        border_style="#D91400",
                        padding=(0, 2),
                    )

                    ui.console.clear()
                    print_fancy_header("NEW CUSTOMER INFORMATION")
                    ui.console.print(input_panel)

                    # Get new information with styled prompts
                    new_details = {}
                    fields = [
                        ("Customer Name", "Full Name"),
                        ("Email Address", "Email"),
                        ("Phone Number", "Phone Number"),
                        ("Order Count", "Number of orders to check"),
                        ("Restaurant", "Restaurant name"),
                    ]

                    for field, prompt in fields:
                        input_value = get_user_input(f"[bold #D91400]{prompt}:[/]", required=False) or ""
                        new_details[field] = input_value
                        # Add row to input table to show progress
                        input_table.add_row(
                            f"[bold]{field}:", f"[green]{input_value}[/]" if input_value else "[dim]-[/]"
                        )
                        ui.console.clear()
                        print_fancy_header("NEW CUSTOMER INFORMATION")
                        ui.console.print(input_panel)

                    # Update app_state with new values
                    app_state.customer_name = new_details["Customer Name"]
                    app_state.customer_email = new_details["Email Address"]
                    app_state.customer_phone = new_details["Phone Number"]
                    app_state.num_orders = new_details["Order Count"]
                    app_state.restaurant_name = new_details["Restaurant"]

                    # Save new session data
                    if save_session_data(app_state):
                        confirm_panel = Panel(
                            "[bold green]✓ Details saved successfully![/]\n\nUpdated information will be shown on the next screen.\n",
                            title="[bold white]SUCCESS",
                            box=DOUBLE,
                            expand=True,
                            border_style="#D91400",
                            padding=(0, 2),
                        )
                        ui.console.print(confirm_panel)
                        time.sleep(1)

                        # Show updated information
                        ui.console.clear()
                        print_fancy_header("UPDATED CUSTOMER DETAILS")
                        show_session_info(app_state, ui)
                        show_message_preview(app_state, ui)
                        return

                elif key in [b"\r", b"\n"]:  # Enter key
                    return

            time.sleep(0.1)


def print_fancy_header(text: str) -> None:
    """Print a modern looking header with proper spacing."""
    ui.header(text)


def get_element_from_text(parent: WebDriver, tag_name: str, text: str, exact: bool = True) -> Optional[WebDriver]:
    """Find element by tag and text."""
    try:
        elements = parent.find_elements(By.TAG_NAME, tag_name)
        for element in elements:
            element_text = element.text
            if exact:
                if element_text == text:
                    return element
            else:
                if text.lower() in element_text.lower():
                    return element
        return None
    except Exception as e:
        ui.status(f"Element search failed: {str(e)} - tag={tag_name}, text={text}", "debug")
        logging.debug(f"Element search failed: {e} - tag={tag_name}, text={text}")
        return None


def get_element_from_attribute(driver: uc.Chrome, tag: str, attribute: str, value: str) -> Optional[uc.WebElement]:
    """Locate a web element based on tag, attribute, and value."""
    try:
        elements = driver.find_elements(By.TAG_NAME, tag)
        for element in elements:
            if element.get_attribute(attribute) == value:
                return element
    except Exception as e:
        logging.error(f"Failed to find element with tag '{tag}', attribute '{attribute}', and value '{value}': {e}")
    return None


def handle_browser_disconnect(driver: uc.Chrome, menu_ui: MenuUI) -> bool:
    """Improved browser disconnection handler with faster checks."""
    try:
        driver.current_url
        return True
    except (WebDriverException, ConnectionError) as e:
        with PRINT_LOCK:
            ui.status("Browser connection lost, attempting recovery...", "warning")
        try:
            # Try to quit gracefully
            try:
                driver.quit()
            except WebDriverException:
                pass

            # Create new driver and restore session
            new_driver = create_driver()
            if auto_login(new_driver):
                driver = new_driver
                return True
            else:
                with PRINT_LOCK:
                    ui.status("Auto-login failed during recovery.", "error")
            menu_ui.handle_login(new_driver)

        except (WebDriverException, ConnectionError) as e:
            ui.status(f"Recovery failed: {str(e)}", "error")
            logging.error(f"Recovery failed: {e}")
        return False


# ====================== TIMEOUT MANAGER ======================
class TimeoutManager:
    """Manages timeouts for the application."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.timeouts = config.get("TIMEOUTS")
        self.wait_times = config.get("WAIT_TIMES")

    def wait(self, wait_type: str):
        """Sleep for a specified wait time."""
        time.sleep(self.wait_times.get(wait_type, self.wait_times["MEDIUM"]))

    def get_timeout(self, timeout_type: str) -> int:
        """Get timeout value for the given type."""
        return self.timeouts.get(timeout_type, self.config.get("TIMEOUTS.DEFAULT_TIMEOUT"))


# ====================== RETRY DECORATOR ======================
def retry_decorator(func):
    """Retry decorator with dynamic backoff."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for attempt in range(config.get("MAX_RETRIES")):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < config.get("MAX_RETRIES") - 1:
                    time.sleep(config.get("WAIT_TIMES.RETRY_DELAY"))
                else:
                    raise

    return wrapper


# ====================== BROWSER AGENT CLASS ======================
class BrowserAgent:
    """Manages browser interactions and provides utility functions."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.timeout_manager = TimeoutManager(config)
        self.logger = logging.getLogger(__name__)

        # Set up WebDriver with timeouts
        chrome_options = uc.ChromeOptions()

        # Set binary location for Brave Browser
        brave_paths = [
            "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe",
            "C:/Program Files (x86)/BraveSoftware/Brave-Browser/Application/brave.exe",
        ]
        for brave_path in brave_paths:
            if os.path.exists(brave_path):
                chrome_options.binary_location = brave_path
                break

        # Add basic options
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--mute-audio")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")

        self.driver = uc.Chrome(options=chrome_options)
        self.driver.set_page_load_timeout(self.timeout_manager.get_timeout("PAGE_LOAD_TIMEOUT"))
        self.driver.implicitly_wait(self.timeout_manager.get_timeout("ELEMENT_TIMEOUT"))

    @retry_decorator
    def wait_for_element(self, by, value):
        """Wait for an element to be present."""
        wait = WebDriverWait(self.driver, self.timeout_manager.get_timeout("ELEMENT_TIMEOUT"))
        return wait.until(EC.presence_of_element_located((by, value)))

    def process_batch(self, items):
        """Process items in a batch."""
        batch_size = min(len(items), config.get("MAX_BATCH_SIZE"))
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            try:
                self._process_items(batch)
                self.timeout_manager.wait("BETWEEN_ORDERS")
            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")

    def scroll_page(self):
        """Scroll to the bottom of the page."""
        self.timeout_manager.wait("SCROLL_PAUSE")
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    def cleanup(self):
        """Clean up the browser."""
        try:
            self.driver.quit()
            time.sleep(self.timeout_manager.get_timeout("CLEANUP_TIMEOUT"))
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def _process_items(self, items: List) -> None:
        """Process a list of items"""
        for item in items:
            self.logger.info(f"Processing item: {item}")
            # Placeholder logic to prevent breaking existing code
            if hasattr(item, "short_id"):
                ui.status(f"Processing #{item.short_id}")
            else:
                ui.status(f"Processing item {item}")
                time.sleep(0.1)


# ====================== ORDER MONITORING ======================
def monitor_for_orders(driver: uc.Chrome, timeout: int = 180) -> bool:
    """Monitor for new orders using a loop."""
    ui.status("No orders found, monitoring...", "info")
    logging.info("No orders found, monitoring...")

    monitor_start = time.time()

    while True:
        # Check overall timeout
        if time.time() - monitor_start > timeout:
            ui.status("Monitoring timed out after 5 minutes", "warning")
            return False

        start_time = time.time()
        try:
            with ui.console.status("[bold white]Monitoring for new orders...[/bold white]", spinner="dots") as status:
                while (time.time() - start_time) < 30:
                    remaining = 30 - (time.time() - start_time)
                    percentage = int(((time.time() - start_time) / 30) * 100)
                    status.update(
                        f"[bold white]Monitoring for new orders... {percentage}% ({remaining:.1f}s remaining)"
                    )

                    if check_keyboard_input() == b"\r":
                        ui.status("Monitoring Interrupted by user.", "warning")
                        return False

                    time.sleep(0.1)

            ui.status("Refreshing page...", "info")
            driver.refresh()  # refreshing in case there were new orders placed

            orders = get_orders(driver, max_orders=5)
            if len(orders) >= 5:
                ui.status("Found at least 5 orders... proceeding", "info")
                logging.info("Found at least 5 orders... proceeding")
                return True

        except Exception as e:
            ui.status(f"Error during monitoring: {str(e)}", "error")
            logging.error(f"Monitoring error: {str(e)}")
            return False


def display_order_summary(orders: List[Order]) -> None:
    """Displays a summary of the orders in a styled panel."""

    total_orders = len(orders)
    orders_to_remove = [o for o in orders if o.amount > 5 and not o.is_canceled]
    removal_count = len(orders_to_remove)
    processed_count = total_orders - removal_count

    ui.status(f"Found {total_orders} orders: {removal_count} to remove, {processed_count} processed/ignored", "info")
    logging.info(f"Found {total_orders} orders: {removal_count} to remove, {processed_count} processed/ignored")
    summary_panel = Panel(
        Text("", justify="left"),
        title="[bold #D91400]Order Summary[/]",
        subtitle=f"[bold #D91400]{total_orders}[/] Orders Found, [bold white]{removal_count}[/] to Remove",
        subtitle_align="right",
        border_style="#D91400",
        padding=(0, 2),
        expand=True,
    )

    # Display the summary panel with a countdown timer
    start_time = time.time()
    # Add countdown before proceeding
    with ui.console.status(
        "[bold #D91400]Processing will start in 10 seconds, press Enter to continue...", spinner="dots"
    ) as status:
        while (time.time() - start_time) < 10:
            remaining = 10 - (time.time() - start_time)
            status.update(f"[bold #D91400]Processing will start in {remaining:.1f} seconds, press Enter to continue...")
            if check_keyboard_input() == b"\r":
                break
            time.sleep(0.1)

    ui.console.print(summary_panel)

    if removal_count > 0:
        ui.console.print(
            Panel(
                Text(", ".join([o.short_id for o in orders_to_remove]), justify="left"),
                title="[bold #D91400]Order Summary[/]",
                subtitle=f"[bold #D91400]{total_orders}[/] Orders Found, [bold white]{removal_count}[/] to Remove",
                subtitle_align="right",
                border_style="#D91400",
                padding=(0, 2),
                expand=True,
            )
        )

    if total_orders == 0:
        ui.status("No orders found.", "info")
    elif removal_count == 0:
        ui.status("No orders found that require removal.", "info")


# ====================== MAIN FUNCTION ======================
def main():
    """Main function with proper error handling."""
    driver = None
    menu_ui = None

    try:
        # Initialize UI components
        menu_ui = MenuUI(app_state)  # Use global app_state

        # Create and configure driver
        driver = create_driver()
        if not driver:
            raise BrowserError("Failed to create browser driver")

        # Initialize processor
        processor = OrderProcessor(driver, config)

        # Run the main processing loop
        asyncio.run(processor.process_orders())

    except BrowserError as e:
        ui.status(f"Browser initialization error: {str(e)}", "error")
        logging.error(f"Browser initialization error: {e}")
        raise
    except WebDriverException as e:
        ui.status(f"WebDriver error: {str(e)}", "error")
        logging.error(f"WebDriver error: {e}")
        if driver:
            handle_browser_disconnect(driver, menu_ui)
        raise
    except Exception as e:
        ui.status(f"Application error: {str(e)}", "error")
        logging.error(f"Application error: {e}", exc_info=True)
        raise
    finally:
        if driver:
            cleanup_driver(driver)


# Create a UI instance
ui = UI()

# Use it to show different types of messages
ui.status("Starting up...", "info")
ui.status("Success!", "success")
ui.status("Warning!", "warning")
ui.status("Error occurred!", "error")

# Show progress
ui.progress(5, 10, "Processing orders")


def main_wrapper():
    """Main wrapper with improved error handling and recovery logic."""
    max_restarts = 5
    ui = ConsoleUI()

    for attempt in range(max_restarts):
        # Calculate restart delay inside the loop where 'attempt' is defined
        restart_delay = pow(5 * attempt, 2) if attempt > 0 else 5  # Exponential backoff with square
        try:
            main()
            break
        except BrowserError as e:
            logging.error(f"Browser error: {e}")
            ui.status(f"Browser error: {str(e)}", "error")
        except LoginError as e:
            logging.error(f"Login error: {e}")
            ui.status(f"Login error: {str(e)}", "error")
        except (RuntimeError, ValueError, ChatError) as e:
            logging.error(f"Unexpected error: {e}")
            ui.status(f"Unexpected error: {str(e)}", "error")

        if attempt < max_restarts - 1:
            ui.status(
                f"Restarting program (attempt {attempt + 1}/{max_restarts})...",
                "warning",
            )
            time.sleep(restart_delay)
        else:
            ui.status("Maximum restart attempts reached", "error")
            break


def test_webdriver():
    ui = UI()  # Using the new UI class
    ui.status("Launching webdriver...", "info")
    try:
        driver = uc.Chrome()
        ui.status("Webdriver launched successfully!", "success")
        return driver
    except Exception as e:
        ui.status(f"Failed to launch webdriver: {e}", "error")
        return None


def test_enhanced_ui():
    ui = EnhancedUI()
    ui.status("Testing Enhanced UI...", "info")

    # Test spinner
    for _ in ui.show_spinner("Loading DoorDash..."):
        time.sleep(2)  # Simulate loading
        break

    # Create some test orders
    test_orders = [
        Order(id="123", short_id="DD123", amount=15.99, status="Ready", is_canceled=False),
        Order(id="124", short_id="DD124", amount=25.50, status="Processing", is_canceled=False),
        Order(id="125", short_id="DD125", amount=10.00, status="Canceled", is_canceled=True),
    ]

    ui.order_summary(test_orders)
    ui.status("✨ UI Test Complete!", "success")


if __name__ == "__main__":
    test_enhanced_ui()
