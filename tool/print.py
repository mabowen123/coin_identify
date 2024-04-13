import datetime


def print_with_timestamp(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("%m-%d%H:%M:%S")
    print(f"[{current_time}]", *args, **kwargs, flush=True)
