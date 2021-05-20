""" default range of waypoints on the map (used for node name binary encoding in states) """


def get_pos_min_max():
    # find boundaries of X and Y coordinates in the graph
    ROW_MIN = 11
    ROW_MAX = 14

    COL_MIN = 0
    COL_MAX = 11
    # TBD for dynamic graphs
    return (ROW_MIN, ROW_MAX), (COL_MIN, COL_MAX)


def get_pos_norms():
    # get range and mins for X and Y coordinates for name encoding
    (row_min, row_max), (col_min, col_max) = get_pos_min_max()
    row_bit = (row_max - row_min).bit_length()
    col_bit = (col_max - col_min).bit_length()
    return (row_bit, col_bit), (row_min, col_min)


def get_node_name_from_pos(row, col):
    (row_b, col_b), _ = get_pos_norms()
    return "{:0{}b}_{:0{}b}".format(row, row_b, col, col_b)


def check_pos_abs_range(pos):
    (r_min, r_max), (c_min, c_max) = get_pos_min_max()
    assert len(pos) == 2 and r_min <= pos[0] <= r_max and c_min <= pos[1] <= c_max, "Pos range error"
    return True


def get_node_name_from_pos_abs(pos, bit_range=None):
    (row_bit, col_bit), (row_min, col_min) = get_pos_norms()
    if bit_range is None:
        row_b, col_b = row_bit, col_bit
    else:
        (row_b, col_b) = bit_range
        # assert row_b < row_bit, "[GymEnv][Parser] ROW encoding bits <{}> not enough < min {}".format(row_b, row_bit)
        # assert col_b < col_bit, "[GymEnv][Parser] COL encoding bits <{}> not enough < min {}".format(col_b, col_bit)
    return "{:0{}b}_{:0{}b}".format(pos[0] - row_min, row_b, pos[1] - col_min, col_b)


def get_node_pos_from_name_abs(name):
    nums = name.split("_")  # name example row_col: "10_1010"
    _, (row_min, col_min) = get_pos_norms()
    return int(nums[0], 2) + row_min, int(nums[1], 2) + col_min


# get the name embedding as a list of numbers
def get_emb_from_name(name):
    nums = name.split("_")
    return [int(row_bit_n) for row_bit_n in nums[0]] + [int(col_bit_n) for col_bit_n in nums[1]]
