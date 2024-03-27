#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def sort_by_template(array_to_sort, template):
    """
    Sort vector 'array_to_sort' according to the order defined by vector 'template'.
    Based on code by Nikita Tiwari.
    https://www.geeksforgeeks.org/sort-array-according-order-defined-another-array/
    """

    if type(array_to_sort) is not np.ndarray:
        array_to_sort = np.array(array_to_sort)
    if type(template) is not np.ndarray:
        template = np.array(template)
    if array_to_sort.ndim > 1 or template.ndim > 1:
        raise ValueError("Both input arrays must be one-dimensional (vectors).")

    m = len(array_to_sort)
    n = len(template)

    # Create a (typically) sorted copy of A and a record of visited elements
    array_temp = np.copy(array_to_sort)
    array_temp.sort()
    visited = np.zeros(m)

    # Create an output array of the same size and type as 'array_to_sort'
    array_ordered = np.empty(array_to_sort.shape, dtype=array_to_sort.dtype)
    i_out = 0

    # Consider all elements of 'template', find them in 'array_temp' and copy to 'a_sorted' in the specified order.
    for i in range(0, n):
        # Find index of the first occurrence of 'template[i]' in 'array_temp' (the copy of 'array_to_sort')
        f = np.where(array_temp == template[i])[0][0]
        # If not present, proceed to next.
        if f == -1:
            continue
        # Copy all occurrences of 'template[i]' to 'a_sorted'.
        j = f
        while j < m and array_temp[j] == template[i]:
            array_ordered[i_out] = array_temp[j]
            i_out += 1
            visited[j] = 1
            j += 1

    # Copy remaining items of 'array_temp' not present in 'template'.
    for i in range(0, m):
        if visited[i] == 0:
            array_ordered[i_out] = array_temp[i]
            i_out += 1

    return array_ordered
