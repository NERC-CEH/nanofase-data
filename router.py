import numpy as np


class Router:
    """The Router class is responsible for using the flow direction raster to route water bodies within the
    model domain, setting variables required by the NanoFASE model, such as inflows, outflows and headwaters."""

    def __init__(self, flow_dir):
        """Initiliase the routing class with the flow direction array."""
        self.flow_dir = flow_dir
        self.grid_mask = self.flow_dir.mask

    def outflow_from_flow_dir(self, x, y):
        """Get the outflow cell reference given the current cell
        reference and a flow direction."""
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
        return xy_out[self.flow_dir[y-1, x-1]]

    def inflows_from_flow_dir(self, x, y):
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
                if self.in_model_domain(i_n, j_n):
                    if self.flow_dir[j_n, i_n] == inflow_flow_dir[(j_n-j, i_n-i)]:
                        inflow_cells.append([i_n+1, j_n+1])
        # Create masked array from the inflow_cells list
        inflow_cells_ma = np.ma.array(np.ma.empty((7,2), dtype=int), mask=True)
        if len(inflow_cells) > 0:       # Only fill if there are inflows
            inflow_cells_ma[0:len(inflow_cells)] = inflow_cells
        return inflow_cells_ma

    def in_model_domain(self, i, j):
        """Check if index [j,i] is in model domain."""
        i_in_domain = 0 <= i < self.grid_mask.shape[1]
        j_in_domain = 0 <= j < self.grid_mask.shape[0]
        not_masked = self.flow_dir[j, i] is not np.ma.masked if (i_in_domain and j_in_domain) else False
        return i_in_domain and j_in_domain and not_masked

    def n_waterbodies_from_inflows(self, x, y, outflow, inflows):
        """Calculate the number of waterbodies from the inflows to the cell."""
        # j, i = y - 1, x - 1
        j_out, i_out = outflow[1] - 1, outflow[0] - 1
        n_inflows = inflows.count(axis=0)[0]        # Count the unmasked elements to get n_inflows
        # If there are no inflows but the outflow is to the model domain, it
        # must be a headwater. Else, number of waterbodies is same as number of inflows
        if n_inflows == 0 and self.in_model_domain(i_out, j_out):
            n_waterbodies = 1
            is_headwater = 1
        else:
            n_waterbodies = n_inflows
            is_headwater = 0
        return n_waterbodies, is_headwater

    # def generate_waterbody_code(self, x, y, outflow, inflows, is_estuary, is_headwater):
    #     """Generates character code of the waterbodies of this cell, of the format
    #     <type><index>(<inflow_point>:<outflow_point>/<n_waterbodies_along_branch>)...
    #     where type = r,e,l,s (river, estuary, lake, sea), index is 1-indexed, inflow
    #     and outflow_side are integers representing the geometrical point of inflow
    #     (0 for centre, 1 for top-left, 2 for top centre, etc.) and n_waterbodies_along_branch
    #     is the number of waterbodies between inflow and outflow."""
    #     outflow_point = self.point_index(outflow[1]-x, outflow[0]-y)
    #     wb_type = 'e' if is_estuary > 0 else 'r'
    #     wb_char = ''
    #     for i, inflow in enumerate(inflows):
    #         if not inflow.mask.any():       # Only if this cell isn't masked
    #             inflow_point = self.point_index(inflow[1]-x, inflow[0]-y)
    #             wb_char = wb_char + '{0}{1}({2}:{3}/01)'.format(wb_type, i+1, inflow_point, outflow_point)
    #     # If this is a headwater, there must be a waterbody with inflow at centre of cell
    #     if is_headwater:
    #         wb_char = '{0}1(0:{1}/01)'.format(wb_type, outflow_point)
    #     return wb_char

    # def point_index(self, di, dj):
    #     point_index = {
    #         (-1, -1): 1,
    #         (0, -1): 8,
    #         (1, -1): 7,
    #         (-1, 0): 2,
    #         (0, 0): 0,
    #         (1, 0): 6,
    #         (-1, 1): 3,
    #         (0, 1): 4,
    #         (1, 1): 5
    #     }
    #     return point_index[dj, di]