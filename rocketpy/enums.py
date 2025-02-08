from enum import Enum


class BaseEnum(Enum):
    def get_members(self):
        return [member for member in self.__class__]


class RocketCoordinateSystemOrientation(BaseEnum):
    TAIL_TO_NOSE = "tail_to_nose"
    NOSE_TO_TAIL = "nose_to_tail"
