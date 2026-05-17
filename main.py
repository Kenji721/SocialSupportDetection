"""
CLI entry point for SSD-2026 Social Support Detection.

Usage:
    python main.py train --task 1
    python main.py train --task 2
    python main.py train --task 3
    python main.py train --task all

    python main.py predict --csv Test_phase_data/test_phase_english.csv --output predictions.csv
    python main.py evaluate --gold Train_Data_SSD26/train-english.csv --pred predictions.csv
"""

import argparse
import sys


def cmd_train(args):
    tasks = [1, 2, 3] if args.task == "all" else [int(args.task)]

    for task_num in tasks:
        print(f"\n{'#' * 60}")
        print(f"# Training Task {task_num}")
        print(f"{'#' * 60}\n")

        if task_num == 1:
            from src.train_task1 import main as train_main
            train_args = argparse.Namespace(
                csv=args.csv,
                text_col=args.text_col,
                label_col="task1",
                max_len=args.max_len,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                save_dir="./task1_model",
            )
        elif task_num == 2:
            from src.train_task2 import main as train_main
            train_args = argparse.Namespace(
                csv=args.csv,
                text_col=args.text_col,
                label_col="task2",
                max_len=args.max_len,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
                save_dir="./task2_model",
            )
        elif task_num == 3:
            from src.train_task3 import main as train_main
            train_args = argparse.Namespace(
                csv=args.csv,
                text_col=args.text_col,
                label_col="task3",
                max_len=args.max_len,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience + 1,  # +1 patience for small dataset
                save_dir="./task3_model",
            )
        else:
            print(f"Unknown task: {task_num}")
            sys.exit(1)

        train_main(train_args)


def cmd_predict(args):
    from src.predict import main as predict_main
    predict_args = argparse.Namespace(
        csv=args.csv,
        text_col=args.text_col,
        task1_model=args.task1_model,
        task2_model=args.task2_model,
        task3_model=args.task3_model,
        output=args.output,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
    predict_main(predict_args)


def cmd_evaluate(args):
    from src.evaluate import main as eval_main
    eval_args = argparse.Namespace(gold=args.gold, pred=args.pred)
    eval_main(eval_args)


def main():
    parser = argparse.ArgumentParser(description="SSD-2026 Social Support Detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a task model")
    train_parser.add_argument("--task", required=True, choices=["1", "2", "3", "all"])
    train_parser.add_argument("--csv", default="data/Train_Data_SSD26/train-english.csv")
    train_parser.add_argument("--text_col", default="text")
    train_parser.add_argument("--max_len", type=int, default=128)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--patience", type=int, default=2)

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Run pipeline inference")
    predict_parser.add_argument("--csv", required=True, help="Path to test CSV")
    predict_parser.add_argument("--text_col", default="text")
    predict_parser.add_argument("--task1_model", default="./task1_model")
    predict_parser.add_argument("--task2_model", default="./task2_model")
    predict_parser.add_argument("--task3_model", default="./task3_model")
    predict_parser.add_argument("--output", default="predictions.csv")
    predict_parser.add_argument("--max_len", type=int, default=128)
    predict_parser.add_argument("--batch_size", type=int, default=32)

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    eval_parser.add_argument("--gold", required=True, help="Gold-standard CSV")
    eval_parser.add_argument("--pred", required=True, help="Predictions CSV")

    # Results subcommand
    subparsers.add_parser("results", help="Show experiment results summary")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "results":
        from src.results import print_summary
        print_summary()


if __name__ == "__main__":
    main()
