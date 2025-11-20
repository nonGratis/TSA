import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import fetch_data

def main():
    # Перевірка аргументів командного рядка
    if len(sys.argv) < 3:
        print("Usage: python main.py <URL> <POLYNOMIAL_DEGREE>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        degree = int(sys.argv[2])
    except ValueError:
        print("Error: Degree must be an integer")
        sys.exit(1)

    print(f"--- Запуск аналізу (Degree: {degree}) ---")

    raw_data = fetch_data(url)
if __name__ == "__main__":
    main()