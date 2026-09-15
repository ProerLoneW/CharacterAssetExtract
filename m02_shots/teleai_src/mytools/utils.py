import os
import time

def print_with_timestamp(*args, **kwargs):
    import time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp} |", *args, **kwargs)

def sleep_for_test():
    if os.environ.get("SLEEP_FOR_TEST", None) is not None:
        if int(os.environ.get("SLEEP_FOR_TEST", 0)) == 1:
            print("sleeping .....")
            time.sleep(1000000)
    print("not test")