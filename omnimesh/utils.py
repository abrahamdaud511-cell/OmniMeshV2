import hashlib
import time
import glob
import os

def get_dir_hash(data_dir):
    hasher = hashlib.md5()
    for file_path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
    return hasher.hexdigest()

def wait_until_safe(get_status_func):
    print("⏸️ System overloaded. Waiting...")
    while True:
        status, cpu, mem = get_status_func()
        if status != "berhenti":
            print(f"✅ System recovered (CPU: {cpu}%, RAM: {mem}%). Resuming...")
            break
        time.sleep(3)
