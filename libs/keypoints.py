"""The AAPose 20-point body layout the pose data is drawn in: the keypoint names, the limbs the
pose images are drawn from, and the index names of the keypoints the pipelines address by
position."""

BODY_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
              "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
              "l_foot", "r_foot"]
# The limbs the pose images are drawn from, as pairs of body keypoints: the same list as
# human_visualization.draw_aapose_new's limbSeq, zero-based. A limb is drawn when both its
# ends reach the draw threshold, so counting them counts what ends up on screen.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
         (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (13, 18), (10, 19)]
HEAD_LIMBS = {(1, 0), (0, 14), (14, 16), (0, 15), (15, 17)}
R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 2, 5, 8, 11
# The body layout carries one foot point per side, the midpoint of that side's two toe
# keypoints; pose2d_utils.split_kp2ds_for_aa averages wholebody 17/18 into the left one and
# 20/21 into the right one, and drops the heels, so these are the only feet there are.
R_ANKLE, L_ANKLE, R_FOOT, L_FOOT = 10, 13, 19, 18
