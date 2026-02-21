"""Allow ``python -m engine.backtest`` to run the backtest pipeline."""

import sys


def _run() -> None:
    from engine.backtest.runner import main
    main()


if __name__ == "__main__":
    _run()
