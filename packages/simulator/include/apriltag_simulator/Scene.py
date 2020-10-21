from .objects import Object3


class Scene:

    def __init__(self, name):
        self._name = name
        self._objects = set()

    def objects(self):
        for obj in self._objects:
            yield obj

    def add(self, obj):
        if not isinstance(obj, Object3):
            raise ValueError('Expected obj to be of type Object3, got %s instead' % str(type(obj)))
        self._objects.add(obj)

    def __iter__(self):
        return iter(self._objects)
