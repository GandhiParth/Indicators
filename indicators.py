from typing import Any, List, Literal, Union

import polars as pl
from polars.selectors import is_selector

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
        if is_selector(columns):
            return self._lf.select(columns).collect_schema().names()
        else:
            return columns

    def sma(
        self, columns: Union[List[str], pl.Expr], window_size: int, _suffix: str = ""
    ) -> "Indicators":
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
            .alias(_suffix + f"{col}_sma_{window_size}")
            for col in columns
        )

        return self

    def awesome_oscillator(
        self, short_window: int = 5, long_window: int = 34, _suffix: str = ""
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
            ).alias(_suffix + f"ao_{short_window}_{long_window}")
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

    def ema(self, columns: Union[List[str], pl.Expr], span: int, _suffix: str = ""):
        """
        Calculate Exponential Moving Average
        """

        columns = self._get_column_names(columns)
        alpha = 2 / (span + 1)

        self._lf = self._lf.with_columns(
            pl.col(col)
            .ewm_mean(alpha=alpha, adjust=False)
            .alias(_suffix + f"{col}_ema_{span}")
            for col in columns
        )

        return self

    def rsi(
        self, columns: Union[List[str], pl.Expr], period: int = 14, _suffix: str = ""
    ):
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
                ).alias(_suffix + f"{col}_rsi_{period}")
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

    def bollinger_bands(
        self,
        columns: Union[List[str], pl.Expr],
        window_size: int = 20,
        num_std_dev: float = 2,
        _suffix: str = "",
    ):
        """
        Calculate Bollinger Bands
        """

        columns = self._get_column_names(columns)

        self._lf = self.sma(columns=columns, window_size=window_size, _suffix="_")._lf
        self._lf = (
            self._lf.with_columns(
                pl.col(col)
                .rolling_std(window_size=window_size)
                .over("symbol")
                .alias(f"_{col}_std_{window_size}")
                for col in columns
            )
            .with_columns(
                (
                    pl.col(f"_{col}_sma_{window_size}")
                    + num_std_dev * pl.col(f"_{col}_std_{window_size}")
                ).alias(_suffix + f"{col}_upper_band_{window_size}_{num_std_dev}")
                for col in columns
            )
            .with_columns(
                (
                    pl.col(f"_{col}_sma_{window_size}")
                    - num_std_dev * pl.col(f"_{col}_std_{window_size}")
                ).alias(_suffix + f"{col}_lower_band_{window_size}_{num_std_dev}")
                for col in columns
            )
            .select(
                pl.exclude(
                    [
                        f"_{col}_{suffix}_{window_size}"
                        for col in columns
                        for suffix in ["std", "sma"]
                    ]
                )
            )
        )

        return self

    def macd(
        self,
        columns: Union[List[str], pl.Expr],
        short_span: int = 12,
        long_span: int = 26,
        signal_span: int = 9,
        _suffix="",
    ):
        """
        Calculates Moving Average Convergence Divergence
        """

        columns = self._get_column_names(columns)

        self._lf = self.ema(columns=columns, span=short_span, _suffix="_")._lf
        self._lf = self.ema(columns=columns, span=long_span, _suffix="_")._lf

        self._lf = self._lf.with_columns(
            (
                pl.col(f"_{col}_ema_{short_span}") - pl.col(f"_{col}_ema_{long_span}")
            ).alias(f"_{col}_macd")
            for col in columns
        )

        self._lf = self.ema(
            columns=[f"_{col}_macd" for col in columns], span=signal_span, _suffix="_"
        )._lf

        self._lf = self._lf.rename(
            {f"__{col}_macd_ema_{signal_span}": f"{col}_signal_line" for col in columns}
        ).select(
            pl.exclude(
                [
                    f"_{col}_ema_{span}"
                    for col in columns
                    for span in [short_span, long_span]
                ]
                + [f"_{col}_macd" for col in columns]
                + [f"__{col}_macd_ema_{signal_span}" for col in columns]
            )
        )

        return self

    def atr(self, period: int = 14, _suffix=""):
        """
        Calculate Average True Range
        """

        self._lf = (
            self._lf.with_columns(
                (pl.col("high") - pl.col("low")).alias("_hl_range"),
                (pl.col("high") - pl.col("close").shift(1)).abs().alias("_hc_range"),
                (pl.col("low") - pl.col("close").shift(1)).abs().alias("_lc_range"),
            )
            .with_columns(
                pl.max_horizontal(["_hl_range", "_hc_range", "_lc_range"]).alias(
                    "_true_range"
                )
            )
            .with_columns(
                pl.col("_true_range")
                .rolling_mean(window_size=period)
                .alias(_suffix + "atr")
            )
            .select(pl.exclude(["_hl_range", "_hc_range", "_lc_range", "_true_range"]))
        )

        return self

    def stochastic_oscillator(self, period: int = 14, _suffix: str = ""):
        """
        Calculate Stochastic Oscillator
        """

        self._lf = (
            self._lf.with_columns(
                (pl.col("close") - pl.col("low"))
                .rolling_min(window_size=period)
                .alias("_numerator"),
                (
                    pl.col("high").rolling_max(window_size=period)
                    - pl.col("low").rolling_min(window_size=period)
                ).alias("_denominator"),
            )
            .with_columns(
                ((pl.col("_numerator") / pl.col("_denominator") * 100)).alias(
                    _suffix + f"stochastic_oscillator_{period}"
                )
            )
            .select(pl.exclude(["_numerator", "_denominator"]))
        )

        return self
