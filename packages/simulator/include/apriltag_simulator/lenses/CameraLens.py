from abc import abstractmethod


class CameraLens:

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def transform(self, x: float, y: float):
        raise NotImplementedError('The lens method should be implemented by the child classes')
