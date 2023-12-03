import random
from typing import List, Tuple


async def standardize_async(column: List[float]) -> List[float]:
    mean = sum(column) / len(column)
    variance = sum((x - mean) ** 2 for x in column) / len(column)
    std_dev = variance ** 0.5
    return [(x - mean) / std_dev for x in column]


async def preprocess_data_async(data: List[List[str]], target_index: int) -> Tuple[List[List[float]], List[float]]:
    features = []
    target = []
    for row in data:
        features.append([float(x) for i, x in enumerate(row) if i != target_index])
        target.append(float(row[target_index]))

    features = list(zip(*features))  # Transpose to standardize each column
    features = [await standardize_async(column) for column in features]
    features = list(zip(*features))  # Transpose back to original structure

    return features, target


async def train_test_split_async(features: List[List[float]], target: List[float], test_size: float = 0.2) ->\
        Tuple[List[List[float]], List[float], List[List[float]], List[float]]:

    combined = list(zip(features, target))
    random.shuffle(combined)
    split = int(len(combined) * test_size)
    test = combined[:split]
    train = combined[split:]
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return list(x_train), list(y_train), list(x_test), list(y_test)

