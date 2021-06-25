from __future__ import annotations

import io
import itertools

import pandas as pd
import matplotlib as mpl

from seaborn.axisgrid import FacetGrid
from seaborn._core.rules import categorical_order, variable_type
from seaborn._core.data import PlotData
from seaborn._core.mappings import GroupMapping, HueMapping
from seaborn._core.scales import (
    ScaleWrapper,
    CategoricalScale,
    DatetimeScale,
    norm_from_scale
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Literal
    from collections.abc import Callable, Generator, Iterable
    from pandas import DataFrame, Series, Index
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.scale import ScaleBase
    from matplotlib.colors import Normalize
    from seaborn._core.mappings import SemanticMapping
    from seaborn._marks.base import Mark
    from seaborn._stats.base import Stat
    from seaborn._core.typing import DataSource, PaletteSpec, VariableSpec


class Plot:

    _data: PlotData
    _layers: list[Layer]
    _mappings: dict[str, SemanticMapping]  # TODO keys as Literal, or use TypedDict?
    _scales: dict[str, ScaleBase]

    _figure: Figure
    _ax: Axes | None
    _facets: FacetGrid | None

    def __init__(
        self,
        data: DataSource = None,
        **variables: VariableSpec,
    ):

        self._data = PlotData(data, variables)
        self._layers = []

        # TODO see notes in _setup_mappings I think we're going to start with this
        # empty and define the defaults elsewhere
        self._mappings = {
            "group": GroupMapping(),
            "hue": HueMapping(),
        }

        # TODO is using "unknown" here the best approach?
        # Other options would be:
        # - None as the value for type
        # - some sort of uninitialized singleton for the object,
        self._scales = {
            "x": ScaleWrapper(mpl.scale.LinearScale("x"), "unknown"),
            "y": ScaleWrapper(mpl.scale.LinearScale("y"), "unknown"),
        }

    def on(self) -> Plot:

        # TODO  Provisional name for a method that accepts an existing Axes object,
        # and possibly one that does all of the figure/subplot configuration

        # We should also accept an existing figure object. This will be most useful
        # in cases where users have created a *sub*figure ... it will let them facet
        # etc. within an existing, larger figure. We still have the issue with putting
        # the legend outside of the plot and that potentially causing problems for that
        # larger figure. Not sure what to do about that. I suppose existing figure could
        # disabling legend_out.

        raise NotImplementedError()
        return self

    def add(
        self,
        mark: Mark,
        stat: Stat | None = None,
        orient: Literal["x", "y", "v", "h"] = "x",  # TODO "auto" as defined by Mark?
        data: DataSource = None,
        **variables: VariableSpec,
    ) -> Plot:

        # TODO we currently need to distinguish between no variables defined for
        # this layer (in which case we inherit the variable specifications from
        # the Plot() constructor) and an empty dictionary of variables, because
        # the latter appears in the faceting context when we join the facet data
        # with each layer's data. That is more evidence that the current way
        # we're handling the facet datajoin is a mess and needs to be reevaluated.
        # Once it is, we can simplify this to just pass the empty dictionary.

        layer_variables = None if not variables else variables
        layer_data = self._data.concat(data, layer_variables)

        if stat is None and mark.default_stat is not None:
            # TODO We need some way to say "do no stat transformation" that is different
            # from "use the default". That's basically an IdentityStat.

            # Default stat needs to be initialized here so that its state is
            # not modified across multiple plots. If a Mark wants to define a default
            # stat with non-default params, it should use functools.partial
            stat = mark.default_stat()

        orient_map = {"v": "x", "h": "y"}
        orient = orient_map.get(orient, orient)  # type: ignore  # mypy false positive?
        mark.orient = orient  # type: ignore  # mypy false positive?
        if stat is not None:
            stat.orient = orient  # type: ignore  # mypy false positive?

        self._layers.append(Layer(layer_data, mark, stat))

        return self

    def _facet(
        self,
        dim: Literal["row", "col"],
        var: VariableSpec = None,
        order: Series | Index | Iterable | None = None,  # TODO alias?
        wrap: int | None = None,
        share: bool | Literal["row", "col"] = True,
        data: DataSource = None,
    ):

        # TODO how to encode the data for this variable?

        # TODO: an issue: `share` is ambiguous because you could configure
        # sharing of both axes for a single dimensional facet. But if we
        # have sharex/sharey in both facet_rows and facet_cols, we will have
        # to handle potentially conflicting specifications. We could also put
        # sharex/sharey in the figure configuration function defaulting to True
        # since without facets, that has no real effect, except we need to
        # sort out how to combine that with the pairgrid functionality.

        self._facetspec[dim] = {
            "order": order,
            "wrap": wrap,
            "share": share,
        }

    # TODO should we have facet_col(var, order, wrap)/facet_row(...)?
    # TODO or facet(dim, var, ...)
    def facet(
        self,
        col: VariableSpec = None,
        row: VariableSpec = None,
        # TODO define our own type alias for order= arguments?
        col_order: Series | Index | Iterable | None = None,
        row_order: Series | Index | Iterable | None = None,
        col_wrap: int | None = None,
        data: DataSource = None,
        **grid_kwargs,  # possibly/probably expose relevant ones
    ) -> Plot:

        # Note: can't pass `None` here or it will undo the `Plot()` def
        variables = {}
        if col is not None:
            variables["col"] = col
        if row is not None:
            variables["row"] = row
        data = self._data.concat(data, variables)

        # TODO raise here if neither col nor row are defined?

        # TODO do we want to allow this method to be optional and create
        # facets if col or row are defined in Plot()? More convenient...

        # TODO another option would be to have this signature be like
        # facet(dim, order, wrap, share)
        # and expect to call it twice for column and row faceting
        # (or have facet_col, facet_row)?

        # TODO what should this data structure be?
        # We can't initialize a FacetGrid here because that will open a figure
        orders = {
            "col": None if col_order is None else list(col_order),
            "row": None if row_order is None else list(row_order),
        }

        facetspec = {}
        for dim in ["col", "row"]:
            if dim in data:
                facetspec[dim] = {
                    "data": data.frame[dim],
                    "order": categorical_order(data.frame[dim], orders[dim]),
                    "name": data.names[dim],
                }

        # TODO accept row_wrap too? If so, move into above logic
        # TODO alternately, change to wrap?
        if "col" in facetspec:
            facetspec["col"]["wrap"] = col_wrap

        facetspec["grid_kwargs"] = grid_kwargs

        self._facetspec = facetspec
        self._facetdata = data  # TODO messy, but needed if variables are added here

        return self

    # TODO map_hue or map_color/map_facecolor/map_edgecolor (or ... all of the above?)
    def map_hue(
        self,
        palette: PaletteSpec = None,
    ) -> Plot:

        # TODO we do some fancy business currently to avoid having to
        # write these ... do we want that to persist or is it too confusing?
        # ALSO TODO should these be initialized with defaults?
        self._mappings["hue"] = HueMapping(palette)
        return self

    def scale_numeric(
        self,
        var: str,
        scale: str | ScaleBase = "linear",
        norm: tuple[float | None, float | None] | Normalize | None = None,
        **kwargs
    ) -> Plot:

        # TODO XXX FIXME matplotlib scales sometimes default to
        # filling invalid outputs with large out of scale numbers
        # (e.g. default behavior for LogScale is 0 -> -10000)
        # This will cause MAJOR PROBLEMS for statistical transformations
        # Solution? I think it's fine to special-case scale="log" in
        # Plot.scale_numeric and force `nonpositive="mask"` and remove
        # NAs after scaling (cf GH2454).
        # And then add a warning in the docstring that the users must
        # ensure that ScaleBase derivatives mask out of bounds data

        # TODO use norm for setting axis limits? Or otherwise share an interface?

        # TODO or separate norm as a Normalize object and limits as a tuple?
        # (If we have one we can create the other)

        # TODO expose parameter for internal dtype achieved during scale.cast?

        if isinstance(scale, str):
            scale = mpl.scale.scale_factory(scale, var, **kwargs)

        if norm is None:
            # TODO what about when we want to infer the scale from the norm?
            # e.g. currently you pass LogNorm to get a log normalization...
            norm = norm_from_scale(scale, norm)
        self._scales[var] = ScaleWrapper(scale, "numeric", norm=norm)
        return self

    def scale_categorical(
        self,
        var: str,
        order: Series | Index | Iterable | None = None,
        formatter: Callable | None = None,
    ) -> Plot:

        # TODO how to set limits/margins "nicely"?
        # TODO similarly, should this modify grid state like current categorical plots?
        # TODO "smart"/data-dependant ordering (e.g. order by median of y variable)

        if order is not None:
            order = list(order)

        scale = CategoricalScale(var, order, formatter)
        self._scales[var] = ScaleWrapper(scale, "categorical")
        return self

    def scale_datetime(self, var) -> Plot:

        scale = DatetimeScale(var)
        self._scales[var] = ScaleWrapper(scale, "datetime")

        # TODO what else should this do?
        # We should pass kwargs to the Datetime case probably.
        # It will be nice to have more control over the formatting of the ticks
        # which is pretty annoying in standard matplotlib.
        # Should datetime data ever have anything other than a linear scale?
        # The only thing I can really think of are geologic/astro plots that
        # use a reverse log scale.

        return self

    def theme(self) -> Plot:

        # TODO We want to be able to use the existing seaborn themeing system
        # to do plot-specific theming
        raise NotImplementedError()
        return self

    def plot(self) -> Plot:

        # TODO note that as currently written this doesn't need to come before
        # _setup_figure, but _setup_figure does use self._scales
        self._setup_scales()

        # === TODO clean series of setup functions (TODO bikeshed names)
        self._setup_figure()

        # ===

        # Abort early if we've just set up a blank figure
        if not self._layers:
            return self

        mappings = self._setup_mappings()

        # scales = self._setup_scales()  TODO?

        for layer in self._layers:

            mappings = {k: v for k, v in mappings.items() if k in layer}

            # TODO very messy but needed to concat with variables added in .facet()
            # Demands serious rethinking!
            if hasattr(self, "_facetdata"):
                layer.data = layer.data.concat(
                    self._facetdata.frame,
                    {v: v for v in ["col", "row"] if v in self._facetdata}
                )
            self._plot_layer(layer, mappings)

        return self

    def _setup_scales(self):

        # TODO one issue here is that we are going to assume all subplots of a
        # figure have the same type of scale. This is potentially problematic if
        # we are not sharing axes ... e.g. we currently can't use displot to
        # show all histograms if some of those histograms need to be categorical.
        # We can decide how much of a problem we are going to consider that to be...
        # It may be better to implement to PairGrid like functionality within Plot
        # and then that can be the "correct" way to mix scales across a figure.

        layers = self._layers
        for var, scale in self._scales.items():
            if scale.type == "unknown" and any(var in layer.data for layer in layers):
                # TODO this is copied from _setup_mappings ... ripe for abstraction!
                all_data = pd.concat(
                    [layer.data.frame.get(var, None) for layer in layers]
                ).reset_index(drop=True)
                scale.type = variable_type(all_data)

    def _setup_figure(self):

        # TODO add external API for parameterizing figure, etc.
        # TODO add external API for parameterizing FacetGrid if using
        # TODO add external API for passing existing ax (maybe in same method)
        # TODO add object that handles the "FacetGrid or single Axes?" abstractions

        if not hasattr(self, "_facetspec"):
            self.facet()  # TODO a good way to activate defaults?

        # TODO use context manager with theme that has been set
        # TODO (or maybe wrap THIS function with context manager; would be cleaner)

        if "row" in self._facetspec or "col" in self._facetspec:

            facet_data = pd.DataFrame()
            facet_vars = {}
            for dim in ["row", "col"]:
                if dim in self._facetspec:
                    name = self._facetspec[dim]["name"]
                    facet_data[name] = self._facetspec[dim]["data"]
                    # TODO FIXME this fails if faceting variables don't have a name
                    # note current relplot also fails, but catplot works...
                    facet_vars[dim] = name
                    if dim == "col":
                        facet_vars["col_wrap"] = self._facetspec[dim]["wrap"]
            kwargs = self._facetspec["grid_kwargs"]
            grid = FacetGrid(facet_data, **facet_vars, pyplot=False, **kwargs)
            grid.set_titles()  # TODO use our own titleing interface?

            self._figure = grid.fig
            self._ax = None
            self._facets = grid

        else:

            self._figure = mpl.figure.Figure()
            self._ax = self._figure.add_subplot()
            self._facets = None

        # TODO we need a new approach here. I think that a flat list of axes
        # objects should be the primary (and possibly only) interface between
        # Plot and the matplotlib Axes it's using. (Possibly the data structure
        # can be list-like with some useful embellishments). We'll handle all
        # the complicated business of setting up a potentially faceted / wrapped
        # / paired figure upstream of that, and downstream will just have the
        # list of Axes.
        #
        # That means we will need some way to map between axes and views on the
        # data (rows for facets and columns for pairs). Then when we are
        # plotting, we will first loop over the axes list, then select the data
        # for each axes, rather than looping over data subsets and finding the
        # corresponding axes. This will let us solve the problem of showing the
        # same plot on all facets. It will also be cleaner.
        #
        # I don't know if we want to package all of the figure setup and mapping
        # between data and axes logic in Plot or if that deserves a separate classs.

        axes_list = list(self._facets.axes.flat) if self._ax is None else [self._ax]
        for ax in axes_list:
            ax.set_xscale(self._scales["x"]._scale)
            ax.set_yscale(self._scales["y"]._scale)

        # TODO good place to do this? (needs to handle FacetGrid)
        obj = self._ax if self._facets is None else self._facets
        for axis in "xy":
            name = self._data.names.get(axis, None)
            if name is not None:
                obj.set(**{f"{axis}label": name})

        # TODO in current _attach, we initialize the units at this point
        # TODO we will also need to incorporate the scaling that (could) be set

    def _setup_mappings(self) -> dict[str, SemanticMapping]:

        layers = self._layers

        # TODO we should setup default mappings here based on whether a mapping
        # variable appears in at least one of the layer data but isn't in self._mappings
        # Source of what mappings to check can be some dictionary of default mappings?

        mappings = {}
        for var, mapping in self._mappings.items():
            if any(var in layer.data for layer in layers):
                all_data = pd.concat(
                    [layer.data.frame.get(var, None) for layer in layers]
                ).reset_index(drop=True)
                scale = self._scales.get(var, None)
                mappings[var] = mapping.setup(all_data, scale)

        return mappings

    def _plot_layer(self, layer, mappings):

        default_grouping_vars = ["col", "row", "group"]  # TODO where best to define?
        grouping_vars = layer.mark.grouping_vars + default_grouping_vars

        data = layer.data
        mark = layer.mark
        stat = layer.stat

        df = self._scale_coords(data.frame)

        if stat is not None:
            df = self._apply_stat(df, grouping_vars, stat)

        df = mark._adjust(df)

        # Our statistics happen on the scale we want, but then matplotlib is going
        # to re-handle the scaling, so we need to invert before handing off
        # Note: we don't need to convert back to strings for categories (but we could?)
        df = self._unscale_coords(df)

        # TODO this might make debugging annoying ... should we create new data object?
        data.frame = df

        generate_splits = self._setup_split_generator(grouping_vars, data, mappings)

        layer.mark._plot(generate_splits, mappings)

    def _apply_stat(
        self, df: DataFrame, grouping_vars: list[str], stat: Stat
    ) -> DataFrame:

        # TODO how can we special-case fast aggregations? (i.e. mean, std, etc.)
        # IDEA: have Stat identify as an aggregator? (Through Mixin or attribute)
        # e.g. if stat.aggregates ...
        stat_grouping_vars = [var for var in grouping_vars if var in df]
        # TODO I don't think we always want to group by the default orient axis?
        # Better to have the Stat declare when it wants that to happen
        if stat.orient not in stat_grouping_vars:
            stat_grouping_vars.append(stat.orient)
        df = (
            df
            .groupby(stat_grouping_vars)
            .apply(stat)
            # TODO next because of https://github.com/pandas-dev/pandas/issues/34809
            .drop(stat_grouping_vars, axis=1, errors="ignore")
            .reset_index(stat_grouping_vars)
            .reset_index(drop=True)  # TODO not always needed, can we limit?
        )
        return df

    def _assign_axes(self, df: DataFrame) -> Axes:
        """Given a faceted DataFrame, find the Axes object for each entry."""
        # TODO the redundancy of self._facets and self._ax screws up type checking
        if self._facets is None:
            assert self._ax is not None  # help mypy
            return self._ax

        df = df.filter(regex="row|col")

        if len(df.columns) > 1:
            zipped = zip(df["row"], df["col"])
            facet_keys = pd.Series(zipped, index=df.index)
        else:
            facet_keys = df.squeeze().astype("category")

        return facet_keys.map(self._facets.axes_dict)

    def _scale_coords(self, df: DataFrame) -> DataFrame:

        coord_df = df.filter(regex="x|y")
        out_df = (
            df
            .drop(coord_df.columns, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        with pd.option_context("mode.use_inf_as_null", True):
            coord_df = coord_df.dropna()

        if self._facets is None:
            assert self._ax is not None  # help mypy
            self._scale_coords_single(coord_df, out_df, self._ax)
        else:
            axes_map = self._assign_axes(df)
            grouped = coord_df.groupby(axes_map, sort=False)
            for ax, ax_df in grouped:
                self._scale_coords_single(ax_df, out_df, ax)
        return out_df

    def _scale_coords_single(
        self, coord_df: DataFrame, out_df: DataFrame, ax: Axes
    ) -> None:

        # TODO modify out_df in place or return and handle externally?

        # TODO this looped through "yx" in original core ... why?
        # for var in "yx":
        #     if var not in coord_df:
        #        continue
        for var, data in coord_df.items():

            # TODO Explain the logic of this method thoroughly
            # It is clever, but a bit confusing!

            axis = var[0]
            axis_obj = getattr(ax, f"{axis}axis")
            scale = self._scales[axis]

            if scale.order is not None:
                data = data[data.isin(scale.order)]

            data = scale.cast(data)
            axis_obj.update_units(categorical_order(data))

            scaled = self._scales[axis].forward(axis_obj.convert_units(data))
            out_df.loc[data.index, var] = scaled

    def _unscale_coords(self, df: DataFrame) -> DataFrame:

        # TODO copied from _scale_coords
        coord_df = df.filter(regex="x|y")
        out_df = (
            df
            .drop(coord_df.columns, axis=1)
            .copy(deep=False)
            .reindex(df.columns, axis=1)  # So unscaled columns retain their place
        )

        for var, col in coord_df.items():
            axis = var[0]
            out_df[var] = self._scales[axis].reverse(coord_df[var])

        return out_df

    def _setup_split_generator(
        self,
        grouping_vars: list[str],
        data: PlotData,
        mappings: dict[str, SemanticMapping],
    ) -> Callable[[], Generator]:

        allow_empty = False  # TODO

        df = data.frame
        # TODO join with axes_map to simplify logic below?

        ax = self._ax
        facets = self._facets

        grouping_vars = [var for var in grouping_vars if var in data]
        if grouping_vars:
            grouped_df = df.groupby(grouping_vars, sort=False, as_index=False)

        levels = {v: m.levels for v, m in mappings.items()}
        if facets is not None:
            for dim in ["col", "row"]:
                if dim in grouping_vars:
                    levels[dim] = getattr(facets, f"{dim}_names")

        grouping_keys = []
        for var in grouping_vars:
            grouping_keys.append(levels.get(var, []))

        iter_keys = itertools.product(*grouping_keys)

        def generate_splits() -> Generator:

            if not grouping_vars:
                yield {}, df.copy(), ax
                return

            for key in iter_keys:

                # Pandas fails with singleton tuple inputs
                pd_key = key[0] if len(key) == 1 else key

                try:
                    df_subset = grouped_df.get_group(pd_key)
                except KeyError:
                    # TODO (from initial work on categorical plots refactor)
                    # we are adding this to allow backwards compatability
                    # with the empty artists that old categorical plots would
                    # add (before 0.12), which we may decide to break, in which
                    # case this option could be removed
                    df_subset = df.loc[[]]

                if df_subset.empty and not allow_empty:
                    continue

                sub_vars = dict(zip(grouping_vars, key))

                # TODO I think we need to be able to drop the faceting vars
                # from a layer and get the same plot on each axes. This is
                # currently not possible. It's going to be tricky because it
                # will require decoupling the iteration over subsets from iteration
                # over facets.

                # TODO can we use axes_map here?
                use_ax: Axes
                if facets is None:
                    assert ax is not None  # help mypy
                    use_ax = ax
                else:
                    row = sub_vars.get("row", None)
                    col = sub_vars.get("col", None)
                    if row is not None and col is not None:
                        use_ax = facets.axes_dict[(row, col)]
                    elif row is not None:
                        use_ax = facets.axes_dict[row]
                    elif col is not None:
                        use_ax = facets.axes_dict[col]
                out = sub_vars, df_subset.copy(), use_ax
                yield out

        return generate_splits

    def show(self) -> Plot:

        # TODO guard this here?
        # We could have the option to be totally pyplot free
        # in which case this method would raise. In this vision, it would
        # make sense to specify whether or not to use pyplot at the initial Plot().
        # Keep an eye on whether matplotlib implements "attaching" an existing
        # figure to pyplot: https://github.com/matplotlib/matplotlib/pull/14024
        # TODO pass kwargs (block, etc.)
        import matplotlib.pyplot as plt
        self.plot()
        plt.show()

        return self

    def save(self) -> Plot:  # or to_file or similar to match pandas?

        raise NotImplementedError()
        return self

    def _repr_png_(self) -> bytes:

        # TODO better to do this through a Jupyter hook?
        # TODO Would like to allow for svg too ... how to configure?
        # TODO We want to skip if the plot has otherwise been shown, but tricky...

        # TODO we need some way of not plotting multiple times
        if not hasattr(self, "_figure"):
            self.plot()

        buffer = io.BytesIO()

        # TODO use bbox_inches="tight" like the inline backend?
        # pro: better results,  con: (sometimes) confusing results
        # Better solution would be to default (with option to change)
        # to using constrained/tight layout.
        self._figure.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()


class Layer:

    # Does this need to be anything other than a simple container for these attributes?
    # Could use a Dataclass I guess?

    def __init__(self, data: PlotData, mark: Mark, stat: Stat = None):

        self.data = data
        self.mark = mark
        self.stat = stat

    def __contains__(self, key: str) -> bool:
        return key in self.data
