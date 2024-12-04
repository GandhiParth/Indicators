from typing import Any, List, Literal

import polars as pl

from utils import validate_columns


class Indicators:
    """
    Calculate Indicators
    """

    def __init__(self, df: pl.DataFrame) -> None:
        """
        Initialize the class
        """
        self._lf = df.lazy().sort("timestamp")

    def collect(self) -> pl.DataFrame:
        """
        Collect the results of the DataFrame
        """
        return self._lf.collect()

    def show_graph(self, optimized: bool = True):
        """
        Show optimized query graph
        """
        return self._lf.show_graph(optimized=optimized)

    def sma(self, columns: List[str], window_size: int) -> "Indicators":
        """
        Calculate Simple Moving Average for the given window_size
        """
        validate_columns(
            required_columns=columns,
            available_columns=self._lf.collect_schema().names(),
        )

        self._lf = self._lf.with_columns(
            pl.col(col)
            .rolling_mean(window_size=window_size)
            .alias(col + f"_sma_{window_size}")
            for col in columns
        )

        return self

    def awesome_oscillator(
        self, short_window: int = 5, long_window: int = 34
    ) -> "Indicators":
        """
        Calculate awesome oscillator
        """
        validate_columns(
            required_columns=["high", "low"],
            available_columns=self._lf.collect_schema().names(),
        )

        self._lf = self._lf.with_columns(
            ((pl.col("high") + pl.col("low")) / 2).alias("_midpoint")
        )
        self._lf = self.sma(columns=["_midpoint"], window_size=short_window)._lf
        self._lf = self.sma(columns=["_midpoint"], window_size=long_window)._lf

        self._lf = self._lf.with_columns(
            (
                pl.col(f"_midpoint_sma_{short_window}")
                - pl.col(f"_midpoint_sma_{long_window}")
            ).alias(f"ao_{short_window}_{long_window}")
        )

        self._lf = self._lf.select(
            pl.exclude(
                "_midpoint",
                f"_midpoint_sma_{short_window}",
                f"_midpoint_sma_{long_window}",
            )
        )

        return self

    def fill_null(
        self,
        columns: List[str],
        value: Any = None,
        method: Literal[
            None, "forward", "backward", "min", "max", "mean", "zero", "one"
        ] = None,
    ):
        if value is not None and method is not None:
            raise ValueError("Either Value or Method can be given")

        if value is None and method is None:
            raise ValueError("Either Value or Method needs to be given")

        validate_columns(
            required_columns=columns,
            available_columns=self._lf.collect_schema().names(),
        )

        if value is not None:
            self._lf = self._lf.with_columns(
                pl.col(col).fill_null(value=value) for col in columns
            )
        else:
            self._lf = self._lf.with_columns(
                pl.col(col).fill_null(strategy=method) for col in columns
            )

        return self

    def kama(self, columns: List[str], n, fastest, slowest):
        """
        Calculate Kaufman's Adaptive Moving Average (KAMA).

        Args:
            df (pl.DataFrame): Input DataFrame.
            column (str): Column for which to calculate KAMA.
            n (int): Lookback period for efficiency ratio.
            fastest (int): Fastest SC period (e.g., 2).
            slowest (int): Slowest SC period (e.g., 30).

        """
