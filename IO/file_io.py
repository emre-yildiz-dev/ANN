import csv
import asyncio
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=5)


async def read_csv_async(filename: str) -> Tuple[List[List[str]], List[str]]:
    loop = asyncio.get_event_loop()
    with open(filename, 'r') as file:
        data = await loop.run_in_executor(executor, lambda: list(csv.reader(file)))
    return data[1:], data[0]  # Data and headers
