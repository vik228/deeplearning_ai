"""
We need to minimize the data movement required during scale ups and scale downs and so consistent hashing is required.
"""
import hashlib
from bisect import bisect, bisect_left, bisect_right


class ConsistentHashing(object):
    def __init__(self):
        self._keys = []
        self._nodes = []
        self._total_slots = 50

    def hash_fn(self, key):
        """
        Create integer equivalent of SHA256 and take modulo from total_slots
        :return: location in hash space
        """

        hsh = hashlib.sha256()
        hsh.update(bytes(key.encode("utf-8")))
        return int(hsh.hexdigest(), 16) % self._total_slots

    def add_node(self, node):
        """
        add node will add a node to a location and return its location
        :return: key of the added node
        """

        key = self.hash_fn(node.host)
        index = bisect(self._keys, key)
        if index > 0 and self._keys[index - 1] == key:
            raise Exception("collision Occurred")
        self._nodes.insert(index, node)
        self._keys.insert(index, key)
        return key

    def remove_node(self, node):
        """
        removes a node and return its key
        :param node:
        :return key:
        """
        key = self.hash_fn(node.host)
        index = bisect_left(self._keys, key)
        if index >= len(self._keys) or self._keys[index] != key:
            raise Exception("Node not found")
        self._nodes.pop(index)
        self._keys.pop(index)
        return key

    def assign(self, item):
        """
        Given the item. The function assigns the item to the node
        :param item:
        :return:
        """
        key = self.hash_fn(item)
        index = bisect_right(self._keys, key) % len(self._keys)
        return self._nodes[index]
