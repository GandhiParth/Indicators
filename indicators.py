from typing import Any, List, Literal, Union

import polars as pl
import polars.selectors as cs

from utils import validate_columns


class Indicators:
    """
    Calculate Indicators
    """

    def __init__(self, df: pl.DataFrame) -> None:
        """
        Initialize the class
        """

        self._lf = df.lazy()
        self._symbol_flag = False

        if "symbol" not in self._lf.collect_schema().names():
            self._symbol_flag = True
            self._lf = self._lf.with_columns(pl.lit("x").alias("symbol"))
        self._lf = self._lf.sort("timestamp")

    def collect(self) -> pl.DataFrame:
        """
        Collect the results of the DataFrame
        """
        if self._symbol_flag:
            self._lf = self._lf.select(pl.exclude("symbol"))
        return self._lf.collect()

    def show_graph(self, optimized: bool = True):
        """
        Show optimized query graph
        """
        return self._lf.show_graph(optimized=optimized)

    def _get_column_names(self, columns: pl.Expr):
        return self._lf.select(columns).collect_schema().names()

    def sma(self, columns: Union[List[str], pl.Expr], window_size: int) -> "Indicators":
        """
        Calculate Simple Moving Average for the given window_size
        """

        columns = self._get_column_names(columns)

        validate_columns(
            required_columns=columns,
            available_columns=self._lf.collect_schema().names(),
        )

        self._lf = self._lf.with_columns(
            pl.col(col)
            .rolling_mean(window_size=window_size)
            .over("symbol")
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
        columns: Union[List[str], pl.Expr],
        value: Any = None,
        method: Literal[
            None, "forward", "backward", "min", "max", "mean", "zero", "one"
        ] = None,
    ):
        if value is not None and method is not None:
            raise ValueError("Either Value or Method can be given")

        if value is None and method is None:
            raise ValueError("Either Value or Method needs to be given")

        columns = self._get_column_names(columns)

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
                pl.col(col).fill_null(strategy=method).over("symbol") for col in columns
            )

        return self

    def ema(self, columns: Union[List[str], pl.Expr], span: int):
        """
        Calculate Exponential Moving Average
        """

        columns = self._get_column_names(columns)
        alpha = 2 / (span + 1)

        self._lf = self._lf.with_columns(
            pl.col(col).ewm_mean(alpha=alpha, adjust=False).alias(f"{col}_ema_{span}")
            for col in columns
        )

        return self

    def rsi(self, columns: Union[List[str], pl.Expr], period: int = 14):
        """
        Calculate Relative Strenght Index
        """
        columns = self._get_column_names(columns)

        self._lf = (
            self._lf.with_columns(
                (pl.col(col) - pl.col(col).shift(1)).alias(f"_{col}_delta")
                for col in columns
            )
            .with_columns(
                pl.when(pl.col(f"_{col}_delta") > 0)
                .then(pl.col(f"_{col}_delta"))
                .otherwise(0)
                .alias(f"_{col}_gain")
                for col in columns
            )
            .with_columns(
                pl.when(pl.col(f"_{col}_delta") < 0)
                .then(pl.col(f"_{col}_delta").abs())
                .otherwise(0)
                .alias(f"_{col}_loss")
                for col in columns
            )
            .with_columns(
                pl.col(f"_{col}_gain")
                .rolling_mean(window_size=period)
                .over("symbol")
                .alias(f"_{col}_avg_gain")
                for col in columns
            )
            .with_columns(
                pl.col(f"_{col}_loss")
                .rolling_mean(window_size=period)
                .over("symbol")
                .alias(f"_{col}_avg_loss")
                for col in columns
            )
            .with_columns(
                (
                    100
                    - (
                        100
                        / (1 + pl.col(f"_{col}_avg_gain") / pl.col(f"_{col}_avg_loss"))
                    )
                ).alias(f"{col}_rsi_{period}")
                for col in columns
            )
            .select(
                pl.exclude(
                    [
                        f"_{col}_{suffix}"
                        for col in columns
                        for suffix in ["delta", "gain", "loss", "avg_gain", "avg_loss"]
                    ]
                )
            )
        )

        return self
