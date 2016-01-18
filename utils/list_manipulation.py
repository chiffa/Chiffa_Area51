

def list_get_indexes(list_, index_list):
    """
    List get indexes: recovers indexes in the list in the provided index list and returns the result in the form of an
    array
    :param list_:
    :param index_list:
    :return:
    """
    if index_list == []:
        return []
    if isinstance(index_list[0], int):
        return np.array([list_[i_] for i_ in index_list])
    if isinstance(index_list[0], bool) or isinstance(index_list[0], np.bool_):
        index_list = np.array(index_list).nonzero()[0]
        return np.array([list_[i_] for i_ in index_list])
    else:
        raise Exception('argument not understood')