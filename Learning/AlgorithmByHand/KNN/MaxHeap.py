# -*- coding: utf-8 -*-
# @time       : 2019-10-22 17:17
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : MaxHeap.py
# @description: 

from time import time
from copy import copy
from random import randint


def get_data(low, high, n_rows, n_cols=None):
    if n_cols is None:
        ret = [randint(low, high) for _ in range(n_rows)]
    else:
        ret = [[randint(low, high) for _ in range(n_cols)] for _ in range(n_rows)]
    return ret


class MaxHeap(object):
    def __init__(self, max_size, fn):
        self.max_size = max_size
        self.fn = fn
        self._items = [None] * max_size
        self.size = 0

    def __str__(self):
        item_values = str([self.fn(self.items[i]) for i in range(self.size)])
        return "Size:%d\nMax size:%d\nItem_values:%s\n" % (self.size, self.max_size, item_values)

    @property
    def items(self):
        return self._items[:self.size]

    @property
    def full(self):
        return self.size == self.max_size

    def value(self, idx):
        item = self._items[idx]
        if item is None:
            ret = -float('inf')
        else:
            ret = self.fn(item)
        return ret

    def add(self, item):
        if self.full:
            if self.fn(item) < self.value(0):
                self._items[0] = item
                self.shift_down(0)
        else:
            self._items[self.size] = item
            self.size += 1
            self.shift_up(self.size - 1)

    def pop(self):
        assert self.size > 0, "Cannot pop item! The MaxHeap is empty!"
        ret = self.items[0]
        self.items[0] = self.items[self.size-1]
        self._items[self.size - 1] = None
        self.size -= 1
        self.shift_down(0)
        return ret

    def shift_up(self, idx):
        assert idx < self.size, "The parameter idx must be less than heap's size"
        parent = (idx - 1) // 2
        while parent >= 0 and self.value(parent) < self.value(idx):
            self._items[parent], self._items[idx] = self._items[idx], self._items[parent]
            idx = parent
            parent = (idx - 1) // 2

    def shift_down(self, idx):
        child = (idx + 1) * 2 - 1
        while child < self.size:
            if child + 1 < self.size and self.value(child + 1) > self.value(child):
                child += 1
            if self.value(idx) < self.value(child):
                self._items[idx], self._items[child] = self._items[child], self._items[idx]
                idx = child
                child = (idx + 1) * 2 - 1
            else:
                break

    def is_valid(self):
        ret = []
        for i in range(1, self.size):
            parent = (i-1) // 2
            ret.append(self.value(parent) >= self.value(i))
        return all(ret)


def exhausted_search(nums, k):
    rets = []
    idxs = []
    key = None
    for _ in range(k):
        val = float('inf')
        for i, num in enumerate(nums):
            if num < val and i not in idxs:
                key = i
                val = num
        idxs.append(key)
        rets.append(val)
    return rets


def main():
    print("Testing MaxHeap...")
    test_times = 100
    run_time_1 = run_time_2 = 0
    low = 0
    high = 1000
    n_rows = 10000
    k = 100
    for _ in range(test_times):
        nums = get_data(low, high, n_rows)

        heap = MaxHeap(k, lambda x: x)
        start = time()
        for num in nums:
            heap.add(num)
        ret1 = copy(heap.items)
        run_time_1 += time() - start

        start = time()
        ret2 = exhausted_search(nums, k)
        run_time_2 += time() - start

        ret1.sort()
        assert ret1 == ret2, "target:%s\nk:%d\nresult1:%s\nresult2:%s\n" % (str(nums), k, str(ret1), str(ret2))
    print("%d tests passed!" % test_times)
    print("Max Heap Search %.2f s" % run_time_1)
    print("Exhausted search %.2f s" % run_time_2)


# main()
