import functools
import pandas as pd
import matplotlib as mpl

import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from seaborn._core.plot import Plot
from seaborn._marks.base import Mark
from seaborn._stats.base import Stat

assert_vector_equal = functools.partial(assert_series_equal, check_names=False)


class MockStat(Stat):

    pass


class MockMark(Mark):

    default_stat = MockStat

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.passed_keys = []
        self.passed_data = []
        self.passed_axs = []

    def _plot_split(self, keys, data, ax, mappings, kws):

        self.passed_keys.append(keys)
        self.passed_data.append(data)
        self.passed_axs.append(ax)


class TestPlot:

    def test_init_empty(self):

        p = Plot()
        assert p._data._source_data is None
        assert p._data._source_vars == {}

    def test_init_data_only(self, long_df):

        p = Plot(long_df)
        assert p._data._source_data is long_df
        assert p._data._source_vars == {}

    def test_init_df_and_named_variables(self, long_df):

        variables = {"x": "a", "y": "z"}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], long_df[col])
        assert p._data._source_data is long_df
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_df_and_mixed_variables(self, long_df):

        variables = {"x": "a", "y": long_df["z"]}
        p = Plot(long_df, **variables)
        for var, col in variables.items():
            if isinstance(col, str):
                assert_vector_equal(p._data.frame[var], long_df[col])
            else:
                assert_vector_equal(p._data.frame[var], col)
        assert p._data._source_data is long_df
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_vector_variables_only(self, long_df):

        variables = {"x": long_df["a"], "y": long_df["z"]}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], col)
        assert p._data._source_data is None
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_vector_variables_no_index(self, long_df):

        variables = {"x": long_df["a"].to_numpy(), "y": long_df["z"].to_list()}
        p = Plot(**variables)
        for var, col in variables.items():
            assert_vector_equal(p._data.frame[var], pd.Series(col))
            assert p._data.names[var] is None
        assert p._data._source_data is None
        assert p._data._source_vars.keys() == variables.keys()

    def test_init_scales(self, long_df):

        p = Plot(long_df, x="x", y="y")
        for var in "xy":
            assert var in p._scales
            assert p._scales[var].type == "unknown"

    def test_add_without_data(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark())
        layer, = p._layers
        assert_frame_equal(p._data.frame, layer.data.frame)

    def test_add_with_new_variable_by_name(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y="y")
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_add_with_new_variable_by_vector(self, long_df):

        p = Plot(long_df, x="x").add(MockMark(), y=long_df["y"])
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_add_with_late_data_definition(self, long_df):

        p = Plot().add(MockMark(), data=long_df, x="x", y="y")
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(layer.data.frame[var], long_df[var])

    def test_add_with_new_data_definition(self, long_df):

        long_df_sub = long_df.sample(frac=.5)

        p = Plot(long_df, x="x", y="y").add(MockMark(), data=long_df_sub)
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x", "y"]
        for var in "xy":
            assert var in layer
            assert_vector_equal(
                layer.data.frame[var], long_df_sub[var].reindex(long_df.index)
            )

    def test_add_drop_variable(self, long_df):

        p = Plot(long_df, x="x", y="y").add(MockMark(), y=None)
        layer, = p._layers
        assert layer.data.frame.columns.to_list() == ["x"]
        assert "y" not in layer
        assert_vector_equal(layer.data.frame["x"], long_df["x"])

    def test_add_stat_default(self):

        p = Plot().add(MockMark())
        layer, = p._layers
        assert layer.stat.__class__ is MockStat

    def test_add_stat_nondefault(self):

        class OtherMockStat(MockStat):
            pass

        p = Plot().add(MockMark(), OtherMockStat())
        layer, = p._layers
        assert layer.stat.__class__ is OtherMockStat

    def test_axis_scale_inference(self, long_df):

        for col, scale_type in zip("zat", ["numeric", "categorical", "datetime"]):
            p = Plot(long_df, x=col, y=col).add(MockMark())
            for var in "xy":
                assert p._scales[var].type == "unknown"
            p._setup_scales()
            for var in "xy":
                assert p._scales[var].type == scale_type

    def test_axis_scale_inference_concatenates(self):

        p = Plot(x=[1, 2, 3]).add(MockMark(), x=["a", "b", "c"])
        p._setup_scales()
        assert p._scales["x"].type == "categorical"

    def test_axis_scale_numeric_as_categorical(self):

        p = Plot(x=[1, 2, 3]).scale_categorical("x", order=[2, 1, 3])
        scl = p._scales["x"]
        assert scl.type == "categorical"
        assert scl.cast(pd.Series([2, 1, 3])).cat.codes.to_list() == [0, 1, 2]

    def test_axis_scale_numeric_as_datetime(self):

        p = Plot(x=[1, 2, 3]).scale_datetime("x")
        scl = p._scales["x"]
        assert scl.type == "datetime"

        numbers = [2, 1, 3]
        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        assert_series_equal(
            scl.cast(pd.Series(numbers)),
            pd.Series(dates, dtype="datetime64[ns]")
        )

    @pytest.mark.xfail
    def test_axis_scale_categorical_as_numeric(self):

        # TODO marked as expected fail because we have not implemented this yet
        # see notes in ScaleWrapper.cast

        strings = ["2", "1", "3"]
        p = Plot(x=strings).scale_numeric("x")
        scl = p._scales["x"]
        assert scl.type == "numeric"
        assert_series_equal(
            scl.cast(pd.Series(strings)),
            pd.Series(strings).astype(float)
        )

    def test_axis_scale_categorical_as_datetime(self):

        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        p = Plot(x=dates).scale_datetime("x")
        scl = p._scales["x"]
        assert scl.type == "datetime"
        assert_series_equal(
            scl.cast(pd.Series(dates, dtype=object)),
            pd.Series(dates, dtype="datetime64[ns]")
        )

    def test_figure_setup_creates_matplotlib_objects(self):

        p = Plot()
        p._setup_figure()
        assert isinstance(p._figure, mpl.figure.Figure)
        assert isinstance(p._ax, mpl.axes.Axes)
