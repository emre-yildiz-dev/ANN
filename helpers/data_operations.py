from typing import List


async def remove_missing_values_async(data: List[List[str]]) -> List[List[str]]:
    return [row for row in data if None not in row and '' not in row]
