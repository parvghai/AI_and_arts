"""Install the desktop runtime locally and relaunch with its Python interpreter."""
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

APP_DIR = Path(__file__).resolve().parent
ENV_DIR = APP_DIR / '.venv'
ENV_PYTHON = ENV_DIR / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
CREATE_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
IMPORT_CHECK = (
    'import torch, mido, tkinter, numpy; '
    'assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 2); '
    'torch.zeros(1)'
)


def runtime_ready(python):
    try:
        result = subprocess.run([str(python), '-c', IMPORT_CHECK],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=CREATE_FLAGS, timeout=60)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def install_runtime():
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title('Vinyl Studio setup')
    root.geometry('540x200')
    root.resizable(False, False)
    status = tk.StringVar(root, 'Preparing the app for its first launch...')
    ttk.Label(root, textvariable=status, wraplength=490, padding=20).pack(fill='x')
    progress = ttk.Progressbar(root, mode='indeterminate', length=490)
    progress.pack(pady=10)
    progress.start()
    ttk.Label(root, text='First setup needs internet and may take a few minutes.').pack(pady=10)
    messages = queue.Queue()
    cancelled = threading.Event()
    completed = []
    log_path = APP_DIR / 'setup.log'

    def run(command, log):
        if cancelled.is_set():
            raise InterruptedError('Setup cancelled. Run main.py again to retry.')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=CREATE_FLAGS, cwd=APP_DIR)
        while process.poll() is None:
            if cancelled.wait(.1):
                process.terminate()
                process.wait()
                raise InterruptedError('Setup cancelled. Run main.py again to retry.')
        if process.returncode:
            raise RuntimeError(f'Setup failed. Check your internet connection and run main.py again.\n\nDetails are in {log_path}')

    def worker():
        try:
            # The operating system releases this lock if a setup process exits.
            with (APP_DIR / '.setup.lock').open('a+b') as lock, log_path.open('a', encoding='utf-8') as log:
                lock.seek(0, os.SEEK_END)
                if not lock.tell():
                    lock.write(b'0')
                    lock.flush()
                messages.put(('status', 'Waiting for any other setup window to finish...'))
                while True:
                    if cancelled.is_set():
                        raise InterruptedError('Setup cancelled.')
                    try:
                        lock.seek(0)
                        if os.name == 'nt':
                            import msvcrt
                            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        else:
                            import fcntl
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError:
                        cancelled.wait(.2)
                if not runtime_ready(ENV_PYTHON):
                    messages.put(('status', 'Creating a private Python environment...'))
                    run([sys.executable, '-m', 'venv', str(ENV_DIR)], log)
                    messages.put(('status', 'Downloading PyTorch and MIDI support...'))
                    run([str(ENV_PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check',
                         '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                         '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
                         '-r', str(APP_DIR / 'requirements.txt')], log)
                    if not runtime_ready(ENV_PYTHON):
                        raise RuntimeError(f'Dependencies could not load. See {log_path} and run main.py again.')
            messages.put(('done', None))
        except Exception as error:
            messages.put(('error', str(error)))

    def close():
        cancelled.set()
        status.set('Cancelling setup...')

    def poll():
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == 'status':
                    status.set(value)
                else:
                    if kind == 'error' and not cancelled.is_set():
                        messagebox.showerror('Setup could not finish', value, parent=root)
                    completed.append(kind == 'done' and not cancelled.is_set())
                    root.destroy()
                    return
        except queue.Empty:
            root.after(100, poll)

    root.protocol('WM_DELETE_WINDOW', close)
    threading.Thread(target=worker, daemon=True).start()
    root.after(100, poll)
    root.mainloop()
    return bool(completed and completed[0])


def ensure_runtime():
    if sys.version_info < (3, 11):
        raise SystemExit('Install 64-bit Python 3.11 or newer, including Tcl/Tk, then run main.py.')
    try:
        import tkinter
    except ImportError:
        raise SystemExit('Python needs Tcl/Tk. Reinstall Python with Tcl/Tk enabled, then run main.py.')
    if runtime_ready(sys.executable):
        return True
    if not runtime_ready(ENV_PYTHON) and not install_runtime():
        raise SystemExit(1)
    result = subprocess.run([str(ENV_PYTHON), str(APP_DIR / 'main.py'), *sys.argv[1:]],
                            cwd=APP_DIR, creationflags=CREATE_FLAGS)
    if result.returncode:
        raise SystemExit(result.returncode)
    return False
