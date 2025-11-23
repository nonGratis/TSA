import sys
import data_handler as dh
import data_analyser as da
import data_vizer as dv

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

    raw_df = dh.fetch_data(url)
    clean_df = dh.prepare_timeseries(raw_df)
if __name__ == "__main__":
    main()