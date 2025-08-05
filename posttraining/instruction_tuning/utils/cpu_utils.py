# Standard Library
import multiprocessing
import os


def get_available_cpu_count() -> int:
    """
    Get the number of available CPU cores.

    Respects BEAKER_ASSIGNED_CPU_COUNT environment variable if set,
    otherwise returns the system CPU count.

    Returns:
        Number of available CPU cores
    """
    return int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))
