import time


def add(a: int, b: int) -> int:
    return a + b


async def add_async(a: int, b: int) -> int:
    time.sleep(2)
    return a + b
