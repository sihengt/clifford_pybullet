from enum import IntEnum

class TransformsIndex(IntEnum):
    POSITION = 0
    ORIENTATION = 1

class LinkStateIndex(IntEnum):
    LINK_WORLD_POSITION = 0                 # cartesian position of COM
    LINK_WORLD_ORIENTATION = 1              # cartesian orientation of COM
    LOCAL_INERTIAL_FRAME_POSITION = 2       # local position offset of inertial frame (center of COM) expressed in URDF link frame
    LOCAL_INERTIAL_FRAME_ORIENTATION = 3    # local orientation offset of the inertial frame
    WORLD_LINK_FRAME_POSITION = 4           # world position of URDF
    WORLD_LINK_FRAME_ORIENTATION = 5        # world orientation of URDF
    WORLD_LINK_LINEAR_VELOCITY = 6          # cartesian world velocity
    WORLD_LINK_ANGULAR_VELOCITY = 7         # cartesian world angular velocity

