import argparse

from spring_engine.workflows import build_data

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog='spring-engine',
        description="Machine learning pipeline for forecasting phenological events from climate data."
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('build-data', help="Build the climate-phenology dataset.")
    subparsers.add_parser('train-model', help="Train the forecasting model.")

    args = parser.parse_args(argv)

    if args.command == 'build-data':
        build_data()
    elif args.command == 'train-model':
        print("Model pipeline is under development.")

if __name__ == "__main__":
    main()