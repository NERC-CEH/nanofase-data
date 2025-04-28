"""
Utilities relating to routing water bodies, such as creating
a flow direction grid from a digital elevation model
"""
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from whitebox.whitebox_tools import WhiteboxTools

from ._errors import RoutingError, whitebox_callback


def in_model_domain(flow_dir, i, j):
    """
    Check if index [j,i] is in the model domain.
    """
    i_in_domain = 0 <= i < flow_dir.mask.shape[1]
    j_in_domain = 0 <= j < flow_dir.mask.shape[0]
    not_masked = flow_dir[j, i] is not np.ma.masked \
        if (i_in_domain and j_in_domain) else False
    return i_in_domain and j_in_domain and not_masked


def outflow_from_flow_dir(flow_dir, x, y):
    """
    Get the outflow cell reference given the current cell
    reference and a flow direction.
    """
    xy_out = {
        1: [x+1, y],
        2: [x+1, y+1],
        4: [x, y+1],
        8: [x-1, y+1],
        16: [x-1, y],
        32: [x-1, y-1],
        64: [x, y-1],
        128: [x+1, y-1]
    }
    return xy_out[flow_dir[y-1, x-1]]


def inflows_from_flow_dir(flow_dir, x, y):
    """Get the inflow cell references given the current cell
    reference and a flow direction."""
    j, i = y - 1, x - 1
    # Flow direction that means the cell with indices inflows[j_n-j, i_n-i]
    # flows into this cell
    inflow_flow_dir = {
        (-1, -1): 2,
        (0, -1): 1,
        (1, -1): 128,
        (-1, 0): 4,
        (0, 0): 0,
        (1, 0): 64,
        (-1, 1): 8,
        (0, 1): 16,
        (1, 1): 32
    }
    inflow_cells = []
    # Loop through the neighbours and check if they're inflows to this cell
    for j_n in range(j-1, j+2):
        for i_n in range(i-1, i+2):
            if in_model_domain(flow_dir, i_n, j_n):
                if flow_dir[j_n, i_n] == inflow_flow_dir[(j_n-j, i_n-i)]:
                    inflow_cells.append([i_n+1, j_n+1])
    # Create masked array from the inflow_cells list
    inflow_cells_ma = np.ma.array(np.ma.empty((7, 2), dtype=int), mask=True)
    if len(inflow_cells) > 0:       # Only fill if there are inflows
        inflow_cells_ma[0:len(inflow_cells)] = inflow_cells
    return inflow_cells_ma


def n_waterbodies_from_inflows(flow_dir, outflow, inflows):
    """
    Calculate the number of waterbodies from the inflows to the cell.
    """
    j_out, i_out = outflow[1] - 1, outflow[0] - 1
    # Count the unmasked elements to get n_inflows
    n_inflows = inflows.count(axis=0)[0]
    # If there are no inflows but the outflow is to the model domain, it
    # must be a headwater. Else, number of waterbodies is same as number of
    # inflows
    if n_inflows == 0 and in_model_domain(flow_dir, i_out, j_out):
        n_waterbodies = 1
        is_headwater = 1
    else:
        n_waterbodies = n_inflows
        is_headwater = 0
    return n_waterbodies, is_headwater


def calculate_flow_dir(dem,
                       res,
                       nodata: Optional[float] = np.nan,
                       treat_nodata_as_drain=True,
                       attempt_to_resolve=False,
                       lonely_cell_flowdir=1,
                       error_if_unresolved=True):
    """
    Calculate the D8 flow direction from a digital elevation model

    Parameters
    ----------
    dem : array-like with dtype=float
        The digital elevation model to calculate flow direction from
    res : length-2 tuple
        The (y, x) resolution of the DEM, used to calculate distances
        for non-square grid cells
    nodata : float, default=np.nan
        The value in `dem` that should be treated as nodata. Defaults
        to NaN
    treat_nodata_as_drain : bool, default=True
        Should nodata cells be treated as a drain, i.e. with lower
        elevation than all other cells? If so, this forces the flow
        direction towards nodata cells by setting their elevation to
        -9999. This is particularly useful where nodata represents
        waterbodies. The `dem` is padded with one pixel of nodata on
        each side in order to calculate flow directions from edge cells,
        and so this flag also affects whether water drains from `dem`
        boundaries. If this is set to False, you may end up with
        unresolved cells
    attempt_to_resolve : bool, default=False
        If the resulting flow direction raster has unresolved cells
        (flow direction = 0), setting `attempt_to_resolve` to True uses
        same flow direction as that from the neighbouring cell with the
        steepest sloe to this cell, effectively presuming the slope carries
        on in the same direction. Warning: this might result in
        neighbouring cells that flow into each other, effectively creating
        "pools" from which flows can't drain
    lonely_cell_flowdir : int or np.nan, default=1
        What flow direction to assign to individual cells surrounded by
        nodata. By default, flow direction is assumed to be eastwards (1).
        If `nodata` or `np.nan`, the cell is removed from the output
        raster. Note that setting to 0 and specifying
        `error_if_unresolved=True` will result in an error if your `dem`
        has lonely cells. There are no further checks on the validity of
        this value, so you can use this to flag lonely cells by setting to
        an arbitrary value
    error_if_unresolved : bool, default=True
        Whether to raise an error if unresolved (flow direction = 0) cells
        are encountered

    Returns
    -------
    np.ma.masked_array(np.int16)
        Masked flow direction array
    """

    # Mappings between D8 codes and directions
    directions = np.array([
        (-1, 1),    # 128 - NE
        (-1, 0),    # 64  - N
        (-1, -1),   # 32  - NW
        (0, -1),    # 16  - W
        (1, -1),    # 8   - SW
        (1, 0),     # 4   - S
        (1, 1),     # 2   - SE
        (0, 1),     # 1   - E
    ])
    d8_codes = np.array([128, 64, 32, 16, 8, 4, 2, 1])
    d8_codes_opposite = np.roll(d8_codes, 4)
    # To make processing easier, we convert nodata values to NaNs for
    # the DEM and -32768 for the flowdir array
    dem = np.where(dem == nodata, np.nan, dem)
    if (lonely_cell_flowdir == nodata) or np.isnan(lonely_cell_flowdir):
        lonely_cell_flowdir = -32768
    # In case there are data right to the edges, we pad with one pixel
    # of NaNs
    dem_padded = np.pad(dem, ((1, 1), (1, 1)), constant_values=np.nan)
    # Get the number of (padded) rows and cols
    nrows, ncols = dem_padded.shape
    # Create a new array to store the (unpadded) flowdir in
    flow_dir = np.full_like(dem, -32768, dtype=np.int16)
    # If `treat_nodata_as_drain` is True, we presume that nodata cells
    # (including the padded region) are water and should be at some
    # negative elevation to force flow to drain into them. If False,
    # they are simply ignored in the flow direction calculation
    dem_adj = np.nan_to_num(dem_padded, nan=-np.inf) \
        if treat_nodata_as_drain else dem_padded

    # Loop over the cells (except the pad)
    for row in range(1, nrows - 1):
        for col in range(1, ncols - 1):
            # Get the value for this cell
            this_elevation = dem_padded[row, col]
            # If this cell is nodata, then we're not interested in the flow
            # direction
            if np.isnan(this_elevation):
                continue
            # Init vars to be update in the loop through neighbouring cells
            max_slope = -np.inf
            min_slope = 0
            heighest_neighbour_direction = 0
            best_direction = 0
            lonely_cell = True
            # Loop through neighbouring cells
            for i, (dr, dc) in enumerate(directions):
                # Get the elevation of this neighbouring cell
                neighbour_elevation = dem_adj[row + dr, col + dc]
                # If the neighbouring cell is nodata and we're not treating
                # nodata as drains, then ignore the influence of this cell
                if np.isnan(neighbour_elevation):
                    continue
                # If neighbouring cell has data, then set the lonely_cell
                # flag to False to say the central cell isn't a single cell
                # surrounded by nodata (including nodata cells treated as
                # drains)
                if neighbour_elevation != -np.inf:
                    lonely_cell = False
                # Difference in elevation and slope
                dz = this_elevation - neighbour_elevation
                distance = np.sqrt((res[0]*dr)**2 + (res[1]*dc)**2)
                slope = dz / distance
                # If slope is steeper than the current maximum, and
                # it's positive (flow *away* from this cell), then
                # set this as the best direction
                if (slope > max_slope) and (slope > 0.0):
                    max_slope = slope
                    best_direction = d8_codes[i]
                elif (slope < min_slope):
                    # If it is a negative slope (this neighbour is higher
                    # than the central cell), store this in case we need it
                    # to try and resolve cells later
                    min_slope = slope
                    heighest_neighbour_direction = d8_codes_opposite[i]

            # Fill the flow direction array. If no candidate has been
            # found then the best_direction is 0 and the cell is unresolved
            flow_dir_ = best_direction

            # If this is a lonely cell, set it to the specified value
            if lonely_cell:
                flow_dir_ = lonely_cell_flowdir
            # If we're meant to, try and resolve this cell and raise an
            # error for unresolved cells
            if (flow_dir_ == 0):
                if attempt_to_resolve:
                    flow_dir_ = heighest_neighbour_direction
                if error_if_unresolved:
                    neighbouring_dem = dem_adj[row-1:row+2, col-1:col+2]
                    raise RoutingError(
                        f"Could not resolve flow direction for cell ({col-1}, {row-1}). "
                        f"DEM for surrounding cells:\n{neighbouring_dem}"
                    )

            # Set the best value for the flow direction
            flow_dir[row - 1, col - 1] = flow_dir_

    # Return the flow direction array as a masked array
    return np.ma.masked_equal(flow_dir, -32768)


def condition_dem(dem_path, config):
    """
    Condition a DEM using Whitebox to remove pits, depressions
    and flat areas
    """
    # If we have been passed a dict of parameters, use these
    if isinstance(config, dict):
        # If a path to save the conditioned DEM has been provided,
        # use this
        if 'save_dem_to_path' in config:
            # Get the path and remove it from the dict so that we
            # don't pass it to the WhiteboxTools function
            dem_path_conditioned = config.pop('save_dem_to_path')
            dem_path_conditioned = Path(dem_path_conditioned).resolve()
            temp_dir = None
        else:
            # If no path has been provided, create a temporary file
            temp_dir = tempfile.TemporaryDirectory()
            dem_path_conditioned = os.path.join(temp_dir.name,
                                                'dem.tif')
        # Use any remainning parameters in the dict as kwargs
        kwargs = config
        # Set dist=10000 if no value given
        kwargs['dist'] = config.get('dist', 10000)
        # Don't use any callback specified as we need to use our
        # own to catch errors
        if 'callback' in kwargs:
            del kwargs['callback']

    wbt = WhiteboxTools()
    # Run the conditioning function and save the resulting DEM
    # to the specified path
    wbt.breach_depressions_least_cost(dem=dem_path,
                                      output=dem_path_conditioned,
                                      callback=whitebox_callback,
                                      **kwargs)
    return dem_path_conditioned, temp_dir
