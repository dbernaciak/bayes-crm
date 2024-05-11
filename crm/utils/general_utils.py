"""General utility functions for the CRM package. """
import time
import hashlib
from tqdm import tqdm
from .numerics import arrival_times


def measure_time_process(func, num_fits, *args, size=100):
    start_time = time.time()
    for _ in tqdm(range(num_fits)):
        times_of_arrival = arrival_times(size)
        _ = func(times_of_arrival, *args)
    return (time.time() - start_time) / num_fits


def measure_time_rej_methods(func, num_fits, *args, size=100):
    start_time = time.time()
    for _ in tqdm(range(num_fits)):
        _ = func(size, *args)
    return (time.time() - start_time) / num_fits


def hash_args(*args, **kwargs):
    # Convert all arguments to string and concatenate them
    arg_str = "".join(str(arg) for arg in args) + "".join(
        f"{k}{v}" for k, v in kwargs.items()
    )

    # Create a SHA256 hash object
    hash_obj = hashlib.sha256()

    # Hash the concatenated string
    hash_obj.update(arg_str.encode("utf-8"))

    # Get the hexadecimal representation of the hash
    hash_hex = hash_obj.hexdigest()

    return hash_hex
