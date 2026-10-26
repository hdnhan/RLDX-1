from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ACTION_HORIZON = 32


human = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_eef_pose",  # 6
            "right_eef_pose",  # 6
            "left_hand_joints",  # 6
            "right_hand_joints",  # 6
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "left_eef_pose",
            "right_eef_pose",
            "left_hand_joints",
            "right_hand_joints",
        ],
        action_configs=[
            # left_eef_pose: chunk expressed relative to the current EEF pose
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_eef_pose
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand_joints
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand_joints
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


vdrd = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head_camera"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_arm",  # 7
            "right_arm",  # 7
            "left_hand",  # 6
            "right_hand",  # 6
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=[
            "left_arm",  # 7
            "right_arm",  # 7
            "left_hand",  # 6
            "right_hand",  # 6
        ],
        action_configs=[
            # left_arm: chunk expressed relative to current joint positions
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(human, EmbodimentTag.NEW_EMBODIMENT)
register_modality_config(vdrd, EmbodimentTag.GENERAL_EMBODIMENT)
