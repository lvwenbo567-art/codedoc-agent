from utils import Calculator, multiply


def main():
    calculator = Calculator()
    result = calculator.add(1, 2)
    print(result)

    value = multiply(3, 4)
    print(value)


if __name__ == "__main__":
    main()