import pickle
import os
from typing import Callable, Dict, Tuple, Any


def _hash(obj: Any) -> str:
    """
    Computes hash of obj. For ints,
    it does not compute hash, rather it
    returns their value. For non-hashable types,
    returns hash of their string representation
    and append 'S' to indicate this

    :param obj: object to be hashed
    :return: hash in hex representation
    """
    try:
        from collections.abc import Hashable
    except ImportError:
        from collections import Hashable

    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, Hashable):
        return hex(hash(obj))
    return hex(hash(str(obj))) + "S"


def _create_name(func: Callable, args: Tuple, kwargs: Dict) -> str:
    """
    Creates name of file to store cache based on
    function name, args and kwargs.

    :param func: Called function
    :param args: its arguments
    :param kwargs: key-word arguments
    :return: str with filename
    """
    name = f"F={func.__name__}__ARGS="
    for arg in args:
        name += f"{_hash(arg)}_"
    name += "_KWARGS="
    for key in sorted(kwargs.keys()):
        val = kwargs[key]
        name += f"{key}-{_hash(val)}_"
    name += ".cache"
    return name


# Path to directory with cached results
CACHE_DIR = "../cache"


def cache_init() -> None:
    """
    Creates directory with cached results
    :return: None
    """
    if not os.path.isdir(CACHE_DIR):
        # If the directory does not exists, create it
        os.mkdir(CACHE_DIR)
    else:
        # otherwise clear it's content
        clear_cache()


def clear_cache() -> None:
    """
    Clear the content of cache directory

    See: https://stackoverflow.com/a/185941
    :return: None
    """
    import shutil
    for file_name in os.listdir(CACHE_DIR):
        file_path = os.path.join(CACHE_DIR, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def cache(function: Callable) -> Callable:
    """
    cache() serves as a decorator for function which takes
    long time to compute and is likely to be used multiple times.
    After first computing of result, stores it to file unique to
    passed function name and arguments. Following calls will result
    in loading the result from FS.

    :param function: function that takes long time to compute
    :return: cached or first-time computed result of function

    Note: it must be possible to pickle.dump() the result
    """
    def load_or_run(*args, **kwargs):
        # Prepare path with cached result
        cache_path = os.path.join(
            CACHE_DIR, _create_name(function, args, kwargs)
        )

        # If the file exists, the result is there
        if os.path.exists(cache_path):
            print(f"-- Loading result from {cache_path}")

            # Open file and return its pickled content
            with open(cache_path, 'rb') as cache_file:
                res = pickle.load(cache_file, encoding="bytes")
            return res

        # If file does not exist, calculate the result and store it
        res = function(*args, **kwargs)

        # Write the result to disk
        with open(cache_path, 'wb') as f:
            print(f"-- Saving result to {cache_path}")
            pickle.dump(res, f)

        # Finally return computed things
        return res
    return load_or_run


def cache_on_the_fly(function: Callable) -> Callable:
    """
    Caches the result of function in dictionary.

    :param function: function doing expensive operation
    :return: cached or first-time computed result of function
    """
    cached_result = {}

    def load_or_run(*args, **kwargs):
        name = _create_name(function, args, kwargs)
        if name not in cached_result:
            print(f"-- Loading from cache")
            cached_result[name] = function(*args, **kwargs)
        return cached_result[name]
    return load_or_run


# Simple demo
if __name__ == "__main__":
    from time import sleep, time

    # Create function that takes long time to compute
    @cache
    def long_time(a: int, b: int):
        # wait 3 secs
        sleep(3)
        return a * b

    # create test CACHE_DIR
    CACHE_DIR = "../cache_test"
    cache_init()

    # Clear cache
    clear_cache()
    print("Clearing cached files")
    print()

    # Run long time and measure time
    start = time()
    result = long_time(3, 4)
    end = time()

    # print results
    print(f"Time: {end - start}s, result is {result}")
    print()

    # Run same again
    start = time()
    result = long_time(3, 4)
    end = time()

    # print results
    print(f"Time: {end - start}s, result is {result}")
    print()

    # Use keyword
    start = time()
    result = long_time(3, b=4)
    end = time()

    # print results
    print(f"Time: {end - start}s, result is {result}")
    print()

    # Use keyword again
    start = time()
    result = long_time(3, b=4)
    end = time()

    # print results
    print(f"Time: {end - start}s, result is {result}")
    print()

    # print content of file
    print(f"Content of {CACHE_DIR}:")
    for filename in os.listdir(CACHE_DIR):
        print(f"=> {filename}")
    print()

    ###########################################
    # demonstration of file names
    long_array = list(range(1000))
    @cache
    def f(arg1, arg2, arg3):
        return 0

    f(long_array, [1, 2, 3], arg3=long_array)
    f(long_array, arg2=[1, 2, 3], arg3=long_array)

    ###########################################
    # on-fly caching
    @cache_on_the_fly
    def g():
        return [1, 2, 3]

    print(g(), flush=True)
    print('===================')
    print(g())
