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

    def ichimoku(
        self, window_one: int = 9, window_two: int = 26, window_three: int = 52
    ):
        """
        Calculate Ichimoku Cloud Components
        """

        self._lf = self._lf.with_columns(
            (
                (
                    pl.col("high").rolling_max(window_size=window_one)
                    + pl.col("low").rolling_min(window_size=window_one)
                )
                / 2
            ).alias("tenkan_sen"),
            (
                (
                    pl.col("high").rolling_max(window_size=window_two)
                    + pl.col("low").rolling_min(window_size=window_two)
                )
                / 2
            ).alias("kijun_sen"),
            (
                (
                    pl.col("high").rolling_max(window_size=window_three)
                    + pl.col("low").rolling_min(window_size=window_three)
                )
                / 2
            )
            .shift(26)
            .alias("senkou_span_b"),
            pl.col("close").shift(-1 * window_two).alias("chikou_span"),
        ).with_columns(
            ((pl.col("tenkan_sen") + pl.col("kijun_sen")) / 2)
            .shift(window_two)
            .alias("senkou_span_a")
        )

        return self

    def vortex(self, period: int = 14) -> "Indicators":
        """
        Calculates Vortex Indicator
        """

        self._lf = (
            self._lf.with_columns(
                pl.max_horizontal(
                    (pl.col("high") - pl.col("low")),
                    (pl.col("high") - pl.col("close").shift(1)).abs(),
                    (pl.col("low") - pl.col("close").shift(1)).abs(),
                ).alias("_tr"),
                (pl.col("high") - pl.col("low").shift(1)).abs().alias("_VM+"),
                (pl.col("low") - pl.col("high").shift(1)).abs().alias("_VM-"),
            )
            .with_columns(
                pl.col("_tr").rolling_sum(window_size=period).alias("_tr_sum"),
                pl.col("_VM+").rolling_sum(window_size=period).alias("_VM+_sum"),
                pl.col("_VM-").rolling_sum(window_size=period).alias("_VM-_sum"),
            )
            .with_columns(
                (pl.col("_VM+_sum") / pl.col("_tr_sum")).alias("+VI"),
                (pl.col("_VM-_sum") / pl.col("_tr_sum")).alias("-VI"),
            )
            .select(
                pl.exclude(["_tr", "_VM+", "_VM-", "_tr_sum", "_VM+_sum", "_VM-_sum"])
            )
        )

        return self

    def trix(self, period: int = 15):
        """
        Calculates the TRIX indicator
        """

        self._lf = self.ema(columns=["close"], span=period, _suffix="_")._lf.rename(
            {f"_close_ema_{period}": "_ema1"}
        )
        self._lf = self.ema(columns=["_ema1"], span=period, _suffix="_")._lf.rename(
            {f"__ema1_ema_{period}": "_ema2"}
        )
        self._lf = self.ema(columns=["_ema2"], span=period, _suffix="_")._lf.rename(
            {f"__ema2_ema_{period}": "_ema3"}
        )

        self._lf = self._lf.with_columns(
            pl.col("_ema3").pct_change(n=1).alias(f"trix_{period}")
        ).select(pl.exclude(["_ema1", "_ema2", "_ema3"]))

        return self

    def mass_index(self, ema_period: int = 9, mi_period: int = 26) -> "Indicators":
        """
        Calculate the Mass Index Indicator
        """

        self._lf = (
            self._lf.with_columns(
                (pl.col("high") - pl.col("low"))
                .ewm_mean(span=ema_period)
                .alias("_ema1")
            )
            .with_columns(pl.col("_ema1").ewm_mean(span=ema_period).alias("_ema2"))
            .with_columns((pl.col("_ema1") / pl.col("_ema2")).alias("_ema_ratio"))
            .with_columns(
                pl.col("_ema_ratio")
                .rolling_sum(window_size=mi_period)
                .alias(f"mass_index_{ema_period}_{mi_period}")
            )
            .select(pl.exclude(["_ema1", "_ema2", "_ema_ratio"]))
        )

        return self

    def psar(
        self, af_start: float = 0.02, af_increment: float = 0.02, af_max: float = 0.02
    ) -> "Indicators":
        """
        Calculates Parabolic Stop and Reverse
        """

        raise NotImplementedError

    def williams_ri(self, lookback: int = 14) -> "Indicators":
        """
        Calculates williams RI
        """

        self._lf = self._lf.with_columns(
            (
                (
                    (pl.col("high").rolling_max(window_size=lookback) - pl.col("close"))
                    / (
                        pl.col("high").rolling_max(window_size=lookback)
                        - pl.col("low").rolling_min(window_size=lookback)
                    )
                )
                * -100
            ).alias(f"williams_%r_{lookback}")
        )
        return self

    def force_index(
        self,
    ) -> "Indicators":
        """
        Calculates Force Index
        """

        self._lf = self._lf.with_columns(
            ((pl.col("close") - pl.col("close").shift(1)) * pl.col("volume")).alias(
                "force_index"
            )
        )

        return self

    def negative_volume_index(self) -> "Indicators":
        """
        Calculates Negative Volume Index
        """

        self._lf = self._lf.with_columns(
            (
                pl.when(pl.col("volume") < pl.col("volume").shift(1))
                .then((pl.col("close") / pl.col("close").shift(1) - 1).cum_sum())
                .otherwise(0)
            ).alias("nvi")
        )

        return self

    def wma(self, columns: Union[List[str], pl.Expr], window_size: int) -> "Indicators":
        """
        Calculate Weighted Moving Average
        """

        raise NotImplementedError

        columns = self._get_column_names(columns=columns)

        weights = pl.Series("weights", range(1, window_size + 1))

        self._lf = self._lf.with_columns(
            (
                pl.col(col).rolling_apply(
                    function=lambda x: (x * weights[-len(x) :]).sum()
                    / weights[-len(x) :].sum(),
                    window_size=window_size,
                )
            ).alias(f"{col}_wma_{window_size}")
            for col in columns
        )

    def donchian_channel(self, look_back: int = 20) -> "Indicators":
        """
        Calculate donchain channel
        """

        self._lf = self._lf.with_columns(
            [
                pl.col("high")
                .rolling_max(window_size=look_back)
                .alias(f"donchian_upper_{look_back}"),
                pl.col("low")
                .rolling_min(window_size=look_back)
                .alias(f"donchian_lower_{look_back}"),
                (
                    (
                        pl.col(f"donchian_upper_{look_back}")
                        + pl.col(f"donchian_lower_{look_back}")
                    )
                    / 2
                ).alias(f"donchian_mid_{look_back}"),
            ]
        )

        return self

    def aroon(self, look_back: int = 14) -> "Indicators":
        """
        Calculate Aroon
        """

        raise NotImplementedError

        self._lf = self._lf.with_columns(
            [
                (
                    100
                    * (
                        look_back
                        - pl.col("high").rolling_apply(
                            function=lambda x: len(x) - x.argmax(),
                            window_size=look_back,
                        )
                    )
                    / look_back
                ).alias(f"aroon_up_{look_back}"),
                (
                    100
                    * (
                        look_back
                        - pl.col("low").rolling_apply(
                            function=lambda x: len(x) - x.argmin(),
                            window_size=look_back,
                        )
                    )
                    / look_back
                ).alias(f"aroon_down_{look_back}"),
            ]
        )

        return self

    def chaikin_money_flow(self, look_back: int = 14) -> "Indicators":
        """ """

        self._lf = self._lf.with_columns(
            (
                (
                    (
                        pl.col("close")
                        - pl.col("low")
                        - (pl.col("high") - pl.col("close"))
                    )
                    / (pl.col("high") - pl.col("low"))
                    * pl.col("volume")
                ).rolling_sum(window_size=look_back)
                / pl.col("volume").rolling_sum(window_size=look_back)
            ).alias(f"cmf_{look_back}")
        )

        return self

    def unlcer_index(
        self, columns: Union[List[str], pl.Expr], look_back: int = 14
    ) -> "Indicators":
        """
        Calculate Ulcer Index
        """

        columns = self._get_column_names(columns)

        self._lf = self._lf.with_columns(
            (
                (
                    (
                        pl.col(col / pl.col(col).rolling_max(window_size=look_back) - 1)
                        ** 2
                    )
                    .rolling_mean(window_size=look_back)
                    .sqrt()
                ).alias(f"{col}_ulcer_index_{look_back}")
                for col in columns
            )
        )

        return self

    def dpo(
        self, columns: Union[List[str], pl.Expr], look_back: int = 20
    ) -> "Indicators":
        """
        Calculate Detrended Price Oscillator
        """

        columns = self._get_column_names(columns)

        offset = (look_back // 2) + 1
        self._lf = self._lf.with_columns(
            (
                pl.col(col)
                - pl.col(col).rolling_mean(window_size=look_back).shift(offset)
            ).alias(f"{col}_dpo_{look_back}")
            for col in columns
        )

        return self

    def kst_oscillator(self, columns: Union[List[str], pl.Expr]) -> "Indicators":
        """ """

        columns = self._get_column_names(columns=columns)

        raise NotImplementedError

        periods = [10, 15, 20, 30]  # Example periods
        weights = [1, 2, 3, 4]  # Example weights
        kst = sum(
            weights[i]
            * pl.col(column).rolling_apply(
                function=lambda x: (x[-1] - x[0]) / x[0], window_size=periods[i]
            )
            for i in range(len(periods))
        )
        return lf.with_columns(kst.alias("kst_oscillator"))

    def ease_of_movement(self):
        """
        Calculate Ease of Movement
        """

        self._lf = self._lf.with_columns(
            (
                (
                    pl.col("high")
                    + pl.col("low")
                    - pl.col("high").shift(1)
                    - pl.col("low").shift(1)
                )
                / pl.col("volume")
            ).alias("ease_of_movement")
        )

        return self

    def true_strength_index(
        self, columns: Union[List[str], pl.Expr], short_period: int, long_preiod: int
    ) -> "Indicators":
        """
        Calculates True Strength Index
        """

        columns = self._get_column_names(columns)

        self._lf = (
            self._lf.with_columns(
                (pl.col(col) - pl.col(col).shift(1)).alias(f"_{col}_delta_price")
                for col in columns
            )
            .with_columns(
                pl.col(f"_{col}_delta_price")
                .ewm_mean(span=short_period)
                .ewm_mean(span=long_preiod)
                .alias(f"_{col}_smooth_delta")
                for col in columns
            )
            .with_columns(
                pl.col(f"_{col}_delta_price")
                .abs()
                .ewm_mean(span=short_period)
                .ewm_mean(span=long_preiod)
                .alias(f"_{col}_smooth_abs_delta")
                for col in columns
            )
            .with_columns(
                (
                    100
                    * pl.col(f"_{col}_smooth_delta")
                    / pl.col(f"_{col}_smooth_abs_delta")
                ).alias(f"{col}_tsi_{short_period}_{long_preiod}")
                for col in columns
            )
            .select(
                pl.exclude(
                    [f"_{col}_smooth{ab}_delta"]
                    for col in columns
                    for ab in ["_abs", ""]
                )
            )
        )

        return self

    def ultimate_oscillator(
        self,
        short: int = 7,
        medium: int = 14,
        long: int = 28,
        short_wt: int = 4,
        medium_wt: int = 2,
        long_wt: int = 2,
    ) -> "Indicators":
        """
        Calculate Ultimate oscialltor
        """

        self._lf = (
            self._lf.with_columns(
                (
                    pl.col("close")
                    - pl.min_horizontal(pl.col("low"), pl.col("close").shift(1))
                ).alias("_bp"),
                (
                    pl.max_horizontal(pl.col("high"), pl.col("close").shift(1))
                    - pl.max_horizontal(pl.col("low"), pl.col("close").shift(1))
                ).alias("_tr"),
            )
            .with_columns(
                (
                    (
                        pl.col("_bp").rolling_mean(window_size=short)
                        / pl.col("_tr").rolling_mean(window_size=short)
                    )
                    * short_wt
                ).alias("_uo_short"),
                (
                    (
                        pl.col("_bp").rolling_mean(window_size=medium)
                        / pl.col("_tr").rolling_mean(window_size=medium)
                    )
                    * medium_wt
                ).alias("_uo_medium"),
                (
                    (
                        pl.col("_bp").rolling_mean(window_size=long)
                        / pl.col("_tr").rolling_mean(window_size=long)
                    )
                    * long_wt
                ).alias("_uo_long"),
            )
            .with_columns(
                (
                    (pl.sum_horizontal("_uo_short", "_uo_medium", "_uo_long") * 100)
                    / (short_wt + medium_wt + long_wt)
                ).alias(f"UO_{short}_{medium}_{long}_{short_wt}_{medium_wt}_{long_wt}")
            )
            .select(pl.exclude(["_bp", "_tr", "_uo_short", "_uo_medium", "_uo_long"]))
        )

        return self

    def keltner_channel(
        self, ema_period: int = 20, atr_period: int = 14, multiplyer: float = 2
    ) -> "Indicators":
        """
        Calculate Keltner Channel
        """

        self._lf = (
            self._lf.with_columns(
                (
                    pl.max_horizontal(pl.col("high"), pl.col("close").shift(1))
                    - pl.min_horizontal(pl.col("low"), pl.col("close").shift(1))
                ).alias("_tr"),
                pl.col("close").ewm_mean(span=ema_period).alias("_middle_band"),
            )
            .with_columns(
                pl.col("_tr").rolling_mean(window_size=atr_period).alias("_atr"),
            )
            .with_columns(
                (pl.col("_middle_band") + pl.col("_atr") * multiplyer).alias(
                    f"KC_Upper_Band_{ema_period}_{atr_period}_{multiplyer}"
                ),
                (pl.col("_middle_band") - pl.col("_atr") * multiplyer).alias(
                    f"KC_Lower_Band_{ema_period}_{atr_period}_{multiplyer}"
                ),
            )
            .select(pl.exclude(["_tr", "_atr", "_middle_band"]))
        )

        return self

    def kst(
        self,
        columns: Union[List[str], pl.Expr],
        roc_periods: tuple = (10, 15, 20, 30),
        sma_periods: tuple = (10, 10, 10, 15),
        weights: tuple = (1, 2, 3, 4),
    ) -> "Indicators":
        """
        Calculate KST Indicators
        """

        raise NotImplementedError

        columns = self._get_column_names(columns)
        assert len(roc_periods) == len(sma_periods) == len(weights)

        rocs = []
        for i, (roc_p, sma_p, w) in enumerate(
            zip(roc_periods, sma_periods, weights), start=1
        ):
            roc = (
                (pl.col(column) - pl.col(column).shift(roc_p))
                / pl.col(column).shift(roc_p)
                * 100
            ).alias(f"ROC_{i}")
            smoothed_roc = (
                pl.col(f"ROC_{i}")
                .rolling_mean(window_size=sma_p)
                .alias(f"Smoothed_ROC_{i}")
            )
            weighted_roc = (pl.col(f"Smoothed_ROC_{i}") * w).alias(f"Weighted_ROC_{i}")
            rocs.extend([roc, smoothed_roc, weighted_roc])

        return (
            lf.with_columns(rocs)
            .with_columns(
                pl.sum(
                    [
                        pl.col(f"Weighted_ROC_{i}")
                        for i in range(1, len(roc_periods) + 1)
                    ]
                ).alias("KST")
            )
            .drop(
                [f"ROC_{i}" for i in range(1, len(roc_periods) + 1)]
                + [f"Smoothed_ROC_{i}" for i in range(1, len(roc_periods) + 1)]
                + [f"Weighted_ROC_{i}" for i in range(1, len(roc_periods) + 1)]
            )
        )

    def stc(
        self,
        columns: Union[List[str], pl.Expr],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        stoch_period: int = 10,
    ) -> "Indicators":
        """
        Calculates schaff cycle trend
        """

        columns = self._get_column_names(columns)
        raise NotImplementedError

        self._lf = (
            self._lf.with_columns(
                (
                    pl.col(col).ewm_mean(span=fast_period)
                    - pl.col(col).ewm_mean(span=slow_period)
                ).alias(f"_{col}_macd_line")
                for col in columns
            )
            .with_columns(
                pl.col(f"_{col}_macd_line")
                .ewm_mean(span=signal_period)
                .alias(f"_{col}_signal_line")
                for col in columns
            )
            .with_columns(
                (
                    (
                        pl.col(f"_{col}_macd_line")
                        - pl.col(f"_{col}_macd_line").rolling_min(
                            window_size=stoch_period
                        )
                    )
                    / (
                        pl.col(f"_{col}_macd_line").rolling_max(
                            window_size=stoch_period
                        )
                        - pl.col(f"_{col}_macd_line").rolling_min(
                            window_size=stoch_period
                        )
                    )
                    * 100
                ).alias(f"{col}_stc_{fast_period}_{slow_period}_{stoch_period}")
                for col in columns
            )
            .select(pl.exclude([]))
        )

    def kama(self):
        raise NotImplementedError

        def kama(lf: pl.LazyFrame, price_col: str, window: int = 10) -> pl.LazyFrame:
            fast_sc = 2 / (2 + 1)
            slow_sc = 2 / (30 + 1)

            return (
                lf.with_columns(
                    (
                        (
                            pl.col(price_col)
                            - pl.col(price_col).shift(window).fill_null(0)
                        )
                        / (
                            pl.col(price_col)
                            - pl.col(price_col)
                            .shift(1)
                            .abs()
                            .rolling_sum(window_size=window)
                        )
                    ).alias("ER")
                )
                .with_columns(
                    ((pl.col("ER") * (fast_sc - slow_sc) + slow_sc) ** 2).alias("SC")
                )
                .with_columns(
                    pl.col(price_col)
                    .ewm_mean(alpha=pl.col("SC"))
                    .alias(f"KAMA_{price_col}")
                )
                .select(pl.all().exclude(["ER", "SC"]))
            )

    def adi(self):
        raise NotImplementedError

        def accumulation_distribution_index(
            lf: pl.LazyFrame,
            high_col: str,
            low_col: str,
            close_col: str,
            volume_col: str,
        ) -> pl.LazyFrame:
            return (
                lf.with_columns(
                    (
                        (
                            (pl.col(close_col) - pl.col(low_col))
                            - (pl.col(high_col) - pl.col(close_col))
                        )
                        / (pl.col(high_col) - pl.col(low_col)).fill_nan(0)
                    ).alias("MFM")
                )
                .with_columns((pl.col("MFM") * pl.col(volume_col)).alias("MFV"))
                .with_columns(pl.col("MFV").cumsum().alias("ADI"))
                .select(pl.all().exclude(["MFM", "MFV"]))
            )

    def mfi(self):
        raise NotImplementedError

        def money_flow_index(
            lf: pl.LazyFrame,
            high_col: str,
            low_col: str,
            close_col: str,
            volume_col: str,
            window: int = 14,
        ) -> pl.LazyFrame:
            return (
                lf.with_columns(
                    # Typical Price
                    (
                        (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
                    ).alias("TP")
                )
                .with_columns(
                    # Raw Money Flow
                    (pl.col("TP") * pl.col(volume_col)).alias("RMF")
                )
                .with_columns(
                    # Positive/Negative Money Flow
                    pl.when(pl.col("TP") > pl.col("TP").shift(1))
                    .then(pl.col("RMF"))
                    .otherwise(0)
                    .rolling_sum(window_size=window)
                    .alias("Positive_MF"),
                    pl.when(pl.col("TP") <= pl.col("TP").shift(1))
                    .then(pl.col("RMF"))
                    .otherwise(0)
                    .rolling_sum(window_size=window)
                    .alias("Negative_MF"),
                )
                .with_columns(
                    # Money Flow Index
                    (
                        100
                        - (100 / (1 + (pl.col("Positive_MF") / pl.col("Negative_MF"))))
                    ).alias("MFI")
                )
                .select(
                    pl.all().exclude(
                        ["TP", "RMF", "Positive_MF", "Negative_MF"]
                    )  # Remove intermediate calculations
                )
            )
