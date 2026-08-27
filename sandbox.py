#Runtime Sandbox for Python Code Execution Phase 6

import os
import sys
import time
import tempfile
import subprocess

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


EXEC_TIMEOUT_SEC = 15       
MEMORY_LIMIT_MB = 512        
POLL_INTERVAL = 0.15         


def _monitor_memory(proc, limit_mb, deadline):
    """
    Παρακολουθεί τη μνήμη της θυγατρικής διεργασίας.
    Επιστρέφει 'MEMORY' αν ξεπεραστεί το όριο, 'TIMEOUT' αν λήξει ο χρόνος,
    None αν η διεργασία τερματίσει κανονικά.
    """
    if not HAS_PSUTIL:
        return None
    try:
        p = psutil.Process(proc.pid)
    except Exception:
        return None

    while proc.poll() is None:
        if time.time() > deadline:
            return "TIMEOUT"
        try:
            rss = p.memory_info().rss
            # Συμπερίληψη θυγατρικών (πιάνει απόπειρες fork bomb)
            for child in p.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except Exception:
                    pass
            if rss > limit_mb * 1024 * 1024:
                return "MEMORY"
        except Exception:
            return None
        time.sleep(POLL_INTERVAL)
    return None


def _kill_tree(proc):
    """Τερματίζει τη διεργασία και όλες τις θυγατρικές της."""
    if HAS_PSUTIL:
        try:
            p = psutil.Process(proc.pid)
            for child in p.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def execute_sandboxed(code_string,
                      timeout=EXEC_TIMEOUT_SEC,
                      memory_mb=MEMORY_LIMIT_MB,
                      workdir=None):
    """
    Εκτελεί κώδικα Python σε απομονωμένη διεργασία με χρονικό όριο και
    όριο μνήμης.

    Επιστρέφει: (output:str, status:str)
      status ∈ {SUCCESS, ERROR, SANDBOX_TIMEOUT, SANDBOX_MEMORY, SANDBOX_ERROR}
    """
    tmp_path = None
    try:
        # Εγγραφή του κώδικα σε προσωρινό αρχείο
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as f:
            f.write(code_string)
            tmp_path = f.name

      
        proc = subprocess.Popen(
            [sys.executable, "-u", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=workdir or os.getcwd(),
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        deadline = time.time() + timeout
        violation = _monitor_memory(proc, memory_mb, deadline)

        if violation == "MEMORY":
            _kill_tree(proc)
            return (f"[Blocked by Runtime Sandbox: Memory limit of "
                    f"{memory_mb} MB exceeded]", "SANDBOX_MEMORY")

        if violation == "TIMEOUT":
            _kill_tree(proc)
            return (f"[Blocked by Runtime Sandbox: Execution timeout of "
                    f"{timeout}s exceeded]", "SANDBOX_TIMEOUT")

        # Αναμονή ολοκλήρωσης με υπόλοιπο χρόνου
        remaining = max(0.1, deadline - time.time())
        try:
            output, _ = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            return (f"[Blocked by Runtime Sandbox: Execution timeout of "
                    f"{timeout}s exceeded]", "SANDBOX_TIMEOUT")

        status = "SUCCESS" if proc.returncode == 0 else "ERROR"
        return (output or "[Executed successfully with no output]"), status

    except Exception as e:
        return f"[Sandbox Error]: {e}", "SANDBOX_ERROR"

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
