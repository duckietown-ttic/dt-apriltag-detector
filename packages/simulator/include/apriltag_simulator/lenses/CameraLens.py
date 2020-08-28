from abc import abstractmethod


class CameraLens:

    def __init__(self, name):
        self._name = name

    @abstractmethod
    def transform(self, point2):
        raise NotImplementedError('The lens method should be implemented by the child classes')