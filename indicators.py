from typing import Any, List, Literal, Union

import polars as pl
from polars.selectors import is_selector

from utils import validate_columns


class Indicators:
    """
    A utility class to calculate financial indicators on a Polars LazyFrame.
    """

    def __init__(self, df: pl.LazyFrame) -> None:
        """
         Initialize the Indicators class with a Polars LazyFrame.

        Args:
            df (pl.LazyFrame): The input LazyFrame containing financial data.
        """

        self._lf: pl.LazyFrame = df
        self._symbol_flag = False

        if "symbol" not in self._lf.collect_schema().names():
            self._symbol_flag = True
            self._lf = self._lf.with_columns(pl.lit("x").alias("symbol"))
        self._lf = self._lf.sort("timestamp")

    def collect(self) -> pl.DataFrame:
        """
        Collect the results of the LazyFrame into a Polars DataFrame.

        Returns:
            pl.DataFrame: The collected DataFrame after all transformations.
        """
        if self._symbol_flag:
            self._lf = self._lf.select(pl.exclude("symbol"))
        return self._lf.collect()

    def show_graph(self, optimized: bool = True):
        """
        Display the query graph for the LazyFrame operations.

        Args:
            optimized (bool): Whether to display the optimized query graph. Defaults to True.

        Returns:
            str: A graphical representation of the query plan.
        """
        if self._symbol_flag:
            self._lf = self._lf.select(pl.exclude("symbol"))

        return self._lf.show_graph(optimized=optimized)

    def get_lazyframe(self) -> pl.LazyFrame:
        """
        Retrieve the current state of the LazyFrame.

        Returns:
            pl.LazyFrame: The LazyFrame with all transformations applied so far.
        """
        return self._lf

    def _get_column_names(self, columns: pl.Expr):
        """
        Resolve column names from a Polars expression or list of column names.

        Args:
            columns (pl.Expr): The column expression or list of column names.

        Returns:
            List[str]: A list of resolved column names.
        """

        return self._lf.select(columns).collect_schema().names()

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
            .over("symbol")
            .alias(_suffix + f"{col}_ema_{span}")
            for col in columns
        )

        return self

    def rsi(
        self, columns: Union[List[str], pl.Expr], period: int = 14, _suffix: str = ""
    ):
        """
        Calculate Relative Strength Index
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
                ).alias(_suffix + f"{col}_upprsier_band_{window_size}_{num_std_dev}")
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

    def _helper_pvo_ppo(
        self,
        col: Literal["close", "volume"],
        short_window: int,
        long_window: int,
        signal_window: int,
    ):
        self._lf = self.ema(columns=[col], span=short_window, _suffix="_")._lf
        self._lf = self.ema(columns=[col], span=long_window, _suffix="_")._lf

        _output_col = "ppo" if col == "close" else "pvo"

        self._lf = self._lf.with_columns(
            (
                (
                    (
                        pl.col(f"_{col}_ema_{short_window}")
                        - pl.col(f"_{col}_ema_{long_window}")
                    )
                    / pl.col(f"_{col}_ema_{long_window}")
                )
                * 100
            ).alias(f"""{_output_col}_{short_window}_{long_window}""")
        )

        self._lf = (
            self.ema(
                columns=[f"""{_output_col}_{short_window}_{long_window}"""],
                span=signal_window,
                _suffix="_",
            )
            ._lf.rename(
                {
                    f"""_{_output_col}_{short_window}_{long_window}_ema_{signal_window}""": f"""{_output_col}_signal_{short_window}_{long_window}"""
                }
            )
            .with_columns(
                (
                    pl.col(f"""{_output_col}_{short_window}_{long_window}""")
                    - pl.col(f"""{_output_col}_signal_{short_window}_{long_window}""")
                ).alias(f"""{_output_col}_histogram_{short_window}_{long_window}""")
            )
            .select(
                pl.exclude(
                    [
                        f"""_{col}_ema_{window}"""
                        for window in [short_window, long_window]
                    ]
                )
            )
        )

        return self

    def ppo(
        self, short_window: int = 12, long_window: int = 26, signal_window: int = 9
    ):
        """
        Calculates the Percentage Price Oscillator
        """
        return self._helper_pvo_ppo(
            col="close",
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
        )

    def pvo(
        self, short_window: int = 12, long_window: int = 26, signal_window: int = 9
    ):
        return self._helper_pvo_ppo(
            col="volume",
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window,
        )

    def roc(self, columns: Union[List[str], pl.Expr], period: int = 10):
        """
        Calculates Rate of Change
        """

        self._lf = self._lf.with_columns(
            (
                (pl.col(col) - pl.col(col).shift(n=period))
                / pl.col(col).shift(n=period)
                * 100
            ).alias(f"{col}_roc_{period}")
            for col in columns
        )

        return self

    def stochastic_rsi(
        self,
        columns: Union[List[str], pl.Expr],
        rsi_period: int = 14,
        stoch_period: int = 14,
    ):
        """
        Stochastic RSI
        """
        columns = self._get_column_names(columns)

        self._lf = self.rsi(columns=columns, period=rsi_period, _suffix="_")._lf

        self._lf = self._lf.with_columns(
            (
                (
                    pl.col(f"_{col}_rsi_{rsi_period}")
                    - pl.col(f"_{col}_rsi_{rsi_period}").rolling_min(
                        window_size=stoch_period
                    )
                )
                / (
                    pl.col(f"_{col}_rsi_{rsi_period}").rolling_max(
                        window_size=stoch_period
                    )
                    - (
                        pl.col(f"_{col}_rsi_{rsi_period}").rolling_max(
                            window_size=stoch_period
                        )
                    )
                )
            ).alias(f"{col}_stoch_rsi_{rsi_period}_{stoch_period}")
            for col in columns
        ).select(pl.exclude([f"_{col}_rsi_{rsi_period}" for col in columns]))

        return self

    def daily_return(
        self, columns: Union[List[str], pl.Expr], _suffix=""
    ) -> "Indicators":
        """
        Calculate the daily return based on the price column.
        """

        columns = self._get_column_names(columns)

        self._lf = self._lf.with_columns(
            ((pl.col(col) - pl.col(col).shift(1)) / pl.col(col).shift(1) * 100)
            .over("symbol")
            .alias(_suffix + f"{col}_daily_return")
            for col in columns
        )

        return self

    def daily_log_return(self, columns: Union[List[str], pl.Expr]) -> "Indicators":
        """
        Calculate Daily Log Return based on price columns.
        """

        columns = self._get_column_names(columns)

        self._lf = self._lf.with_columns(
            (pl.col(col) / pl.col(col).shift(1))
            .log()
            .over("symbol")
            .alias(f"{col}_daily_log_return")
            for col in columns
        )

        return self

    def cumulative_returns(self, columns: Union[List[str], pl.Expr]) -> "Indicators":
        """
        Calculates the Cumulative Returns
        """

        columns = self._get_column_names(columns)

        self._lf = self.daily_return(columns=columns, _suffix="_")._lf
        self._lf = self._lf.with_columns(
            (pl.col(f"_{col}_daily_return") + 1)
            .cum_prod()
            .over("symbol")
            .alias(f"{col}_cumulative_return")
            for col in columns
        ).select(pl.exclude([f"_{col}_daily_return" for col in columns]))

        return self

    # def weighted_moving_average(
    #     self, columns: Union[List[str], pl.Expr], window_size: int
    # ):
    #     """
    #     Calculate the Weighted Moving Average (WMA) for a given column.
    #     """

    #     columns = self._get_column_names(columns)

    #     weights = pl.Series("weights", [i for i in range(1, window_size + 1)])

    #     self_lf = self._lf.with_columns(
    #         (
    #             pl.col(col).rolling_apply(
    #                 lambda values: (values * weights).sum() / weights.sum(),
    #                 window_size=window_size,
    #                 min_periods=window_size,
    #             )
    #         )
    #         .over("symbol")
    #         .alias(f"{col}_wma_{window_size}")
    #         for col in columns
    #     )

    def adx(self, period: int = 14):
        """
        Calculates Average Direction Index
        """

        self._lf = (
            self._lf.with_columns(
                pl.max_horizontal(
                    pl.col("high") - pl.col("close"),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ).alias("_tr"),
                (pl.col("high") - pl.col("high").shift(1)).alias("_high"),
                (pl.col("low").shift(1) - pl.col("low")).alias("_low"),
            )
            .with_columns(
                pl.when(pl.col("_high") > 0)
                .then(pl.col("_high"))
                .otherwise(0)
                .alias("_+dm"),
                pl.when(pl.col("_low") > 0)
                .then(pl.col("_low"))
                .otherwise(0)
                .alias("_-dm"),
            )
            .with_columns(
                pl.col("_tr").rolling_mean(window_size=period).alias("_smooth_tr"),
                pl.col("_+dm").rolling_mean(window_size=period).alias("_smooth_+dm"),
                pl.col("_-dm").rolling_mean(window_size=period).alias("_smooth_-dm"),
            )
            .with_columns(
                (pl.col("_smooth_+dm") / pl.col("_smooth_tr") * 100).alias("_+di"),
                (pl.col("_smooth_-dm") / pl.col("_smooth_tr") * 100).alias("_-di"),
            )
            .with_columns(
                (
                    (pl.col("_+di") - pl.col("_-di")).abs()
                    / (pl.col("_+di") + pl.col("_-di"))
                    * 100
                ).alias("_dx")
            )
            .with_columns(pl.col("_dx").rolling_mean(window_size=period).alias("adx"))
            .select(
                pl.exclude(
                    [
                        "_dx",
                        "_+di",
                        "_-di",
                        "_smooth_-dm",
                        "_smooth_+dm",
                        "_smooth_tr",
                        "_-dm",
                        "_+dm",
                        "_high",
                        "_low",
                        "_tr",
                    ]
                )
            )
        )

        return self

    def aroon(self, period: int = 14):
        """
        Calculate the Aroon Up and Aroon Down Indicators
        """

        raise NotImplementedError

    def cci(self, period: int = 14):
        """
        Calculates the Comodity Channel Index
        """

        self._lf = (
            self._lf.with_columns(
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias(
                    "_typical_price"
                )
            )
            .with_columns(
                pl.col("_typical_price")
                .rolling_mean(window_size=period)
                .alias("_sma_tp")
            )
            .with_columns(
                (pl.col("_typical_price") - pl.col("_sma_tp"))
                .abs()
                .rolling_mean(window_size=period)
                .alias("_mean_deviation")
            )
            .with_columns(
                (
                    (pl.col("_typical_price") - pl.col("_sma_tp"))
                    / (0.015 * pl.col("_mean_deviation"))
                ).alias("CCI")
            )
            .select(pl.exclude(["_mean_deviation", "_sma_tp", "_typical_price"]))
        )

        return self

    def vwap(
        self,
    ) -> "Indicators":
        """
        Calculates Volume Weighted Average Price
        """

        self._lf = (
            self._lf.with_columns(
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias(
                    "_typical_price"
                )
            )
            .with_columns(
                (pl.col("_typical_price") * pl.col("volume")).alias("_tp_vol")
            )
            .with_columns(
                pl.col("_tp_vol").cum_sum().alias("_cumsum_tp_vol"),
                pl.col("volume").cum_sum().alias("_cumsum_vol"),
            )
            .with_columns(
                (pl.col("_cumsum_tp_vol") / pl.col("_cumsum_vol")).alias("vwap")
            )
            .select(
                pl.exclude(
                    ["_typical_price", "_tp_vol", "_cumsum_tp_vol", "_cumsum_vol"]
                )
            )
        )

        return self

    def vpt(self) -> "Indicators":
        """
        Calculates Volume Price Trend
        """

        self._lf = (
            self._lf.with_columns(pl.col("close").pct_change(n=1).alias("_pct_change"))
            .with_columns(
                (pl.col("_pct_change") * pl.col("volume")).alias("_vpt_change")
            )
            .with_columns(pl.col("_vpt_change").cum_sum().alias("vpt"))
            .select(pl.exclude(["_pct_change", "_vpt_change"]))
        )

        return self

    def obv(self):
        """
        Calculates OnBalance Volume
        """

        self._lf = (
            self._lf.with_columns(
                pl.when(pl.col("close").diff() > 0)
                .then(1)
                .otherwise(pl.when(pl.col("close").diff() < 0).then(-1).otherwise(0))
                .alias("_direction")
            )
            .with_columns(
                (pl.col("_direction") * pl.col("volume")).alias("_obv_change")
            )
            .with_columns(pl.col("_obv_change").cum_sum().alias("obv"))
            .select(pl.exclude(["_direction", "_obv_change"]))
        )

        return self

    def nvi(self) -> "Indicators":
        """
        calculates Negative Volume Index
        """

        self._lf = self._lf.with_columns(
            pl.col("close").pct_change().alias("_pct_change")
        )

        raise NotImplementedError
