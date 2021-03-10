import torch


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
        >>> d.c = [[7], [8]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5], "c": [7]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def __repr__(self):
        _d = {k: dict.__getitem__(self, k) for k in self.keys()}
        return _d.__repr__()

    def append(self, other):
        """ Append another DictList with one element """
        if not isinstance(other, DictList):
            other = DictList(other)
        for key in other:
            if key not in self:
                self.__setattr__(key, [])
            existing = self.__getattr__(key)
            existing.append(other.__getattr__(key))

    def __add__(self, other):
        """ Concat """
        if not isinstance(other, DictList):
            other = DictList(other)
        res = {}
        for key in other:
            if key not in self:
                self.__setattr__(key, [])
            existing = self.__getattr__(key)
            res[key] = existing + other.__getattr__(key)
        return DictList(res)

    def apply(self, func):
        """ Apply the same func to each element """
        for key, val in self.items():
            self.__setattr__(key, func(val))

    def report(self):
        for key, val in self.items():
            print(key, val.shape)


class Metric:
    """ A general metric object for recording mean """
    def __init__(self, fields):
        self.data = {name: 0 for name in fields}
        self.count = {name: 0 for name in fields}

    def mean(self):
        return {name: self.data[name] / self.count[name] if self.count[name] != 0 else 0
                for name in self.data}

    def reset(self):
        self.data = {name: 0 for name in self.data}
        self.count = {name: 0 for name in self.data}

    def accumulate(self, stats, counts):
        if isinstance(stats, DictList):
            stats = {k: v for k, v in stats.items()}
        for name in stats:
            if self.data.get(name, None) is None:
                self.data[name] = 0
                self.count[name] = 0

            if isinstance(stats[name], torch.Tensor):
                self.data[name] += stats[name].item() * counts
            elif isinstance(stats[name], float):
                self.data[name] += stats[name] * counts
            self.count[name] += counts

    def __str__(self):
        result = self.mean()
        return "\t".join(['{}:{:.4f}'.format(key, val) for key, val in result.items()])
