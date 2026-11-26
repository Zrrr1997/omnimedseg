"""Progress indicators for long-running operations."""
import time
from contextlib import contextmanager
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@contextmanager
def show_progress(description, total=None, unit='items'):
    """
    Context manager for showing progress.
    
    Usage:
        with show_progress("Processing files", total=100) as pbar:
            for i in range(100):
                time.sleep(0.01)
                pbar.update(1)
    """
    pbar = tqdm(total=total, desc=description, unit=unit)
    try:
        yield pbar
    finally:
        pbar.close()


def log_step(step_name):
    """
    Decorator to log function execution.
    
    Usage:
        @log_step("Downloading dataset")
        def download():
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.info(f"▶️ Starting: {step_name}")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"✅ Completed: {step_name} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                logger.error(f"❌ Failed: {step_name} - {e}")
                raise
        return wrapper
    return decorator


class ProgressTracker:
    """Track progress across multiple steps."""
    
    def __init__(self, total_steps, description="Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.pbar = tqdm(total=total_steps, desc=description)
    
    def step(self, message=None):
        """Advance progress by one step."""
        self.current_step += 1
        if message:
            self.pbar.set_postfix_str(message)
        self.pbar.update(1)
    
    def close(self):
        """Close the progress bar."""
        self.pbar.close()
