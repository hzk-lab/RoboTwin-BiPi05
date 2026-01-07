from ._base_task import Base_Task
from .utils import *
import sapien
from copy import deepcopy


class shake_dual_bottles(Base_Task):

    def setup_demo(self, is_test=False, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.id_list = [i for i in range(20)]
        
        # Select two different bottle IDs
        self.bottle1_id = np.random.choice(self.id_list)
        remaining_ids = [i for i in self.id_list if i != self.bottle1_id]
        self.bottle2_id = np.random.choice(remaining_ids)

        # Left bottle (for left arm)
        self.bottle1 = rand_create_actor(
            self,
            xlim=[-0.25, -0.05],
            ylim=[-0.15, -0.05],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
            qpos=[0, 0, 1, 0],
            convex=True,
            model_id=self.bottle1_id,
        )
        self.bottle1.set_mass(0.01)

        # Right bottle (for right arm)
        self.bottle2 = rand_create_actor(
            self,
            xlim=[0.05, 0.25],
            ylim=[-0.15, -0.05],
            modelname="001_bottle",
            rotate_rand=True,
            rotate_lim=[0, 0, np.pi / 4],
            qpos=[0, 0, 1, 0],
            convex=True,
            model_id=self.bottle2_id,
        )
        self.bottle2.set_mass(0.01)

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

        self.add_prohibit_area(self.bottle1, padding=0.05)
        self.add_prohibit_area(self.bottle2, padding=0.05)

    def play_once(self):
        # Assign arms: left arm for bottle1, right arm for bottle2
        bottle1_arm_tag = ArmTag("left")
        bottle2_arm_tag = ArmTag("right")

        # Simultaneously grasp both bottles
        self.move(
            self.grasp_actor(self.bottle1, arm_tag=bottle1_arm_tag, pre_grasp_dis=0.1),
            self.grasp_actor(self.bottle2, arm_tag=bottle2_arm_tag, pre_grasp_dis=0.1),
        )

        # Simultaneously lift both bottles and rotate to target orientation
        target_quat = [0.707, 0, 0, 0.707]
        self.move(
            self.move_by_displacement(arm_tag=bottle1_arm_tag, z=0.1, quat=target_quat),
            self.move_by_displacement(arm_tag=bottle2_arm_tag, z=0.1, quat=target_quat),
        )

        # Prepare two shaking orientations by rotating around y-axis
        quat1 = deepcopy(target_quat)
        quat2 = deepcopy(target_quat)
        
        # First shake rotation (7π/8 around y-axis)
        y_rotation = t3d.euler.euler2quat(0, (np.pi / 8) * 7, 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat1)
        quat1 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        # Second shake rotation (-7π/8 around y-axis)
        y_rotation = t3d.euler.euler2quat(0, -7 * (np.pi / 8), 0)
        rotated_q = t3d.quaternions.qmult(y_rotation, quat2)
        quat2 = [-rotated_q[1], rotated_q[0], rotated_q[3], -rotated_q[2]]

        # Perform shaking motion three times (alternating between two orientations)
        for _ in range(3):
            # Move up with first shaking orientation (simultaneously)
            self.move(
                self.move_by_displacement(arm_tag=bottle1_arm_tag, z=0.05, quat=quat1),
                self.move_by_displacement(arm_tag=bottle2_arm_tag, z=0.05, quat=quat1),
            )
            # Move down with second shaking orientation (simultaneously)
            self.move(
                self.move_by_displacement(arm_tag=bottle1_arm_tag, z=-0.05, quat=quat2),
                self.move_by_displacement(arm_tag=bottle2_arm_tag, z=-0.05, quat=quat2),
            )

        # Return to original grasp orientation (simultaneously)
        self.move(
            self.move_by_displacement(arm_tag=bottle1_arm_tag, quat=target_quat),
            self.move_by_displacement(arm_tag=bottle2_arm_tag, quat=target_quat),
        )

        self.info["info"] = {
            "{A}": f"001_bottle/base{self.bottle1_id}",
            "{B}": f"001_bottle/base{self.bottle2_id}",
        }
        return self.info

    def check_success(self):
        # Check that both bottles are lifted above the target height
        bottle1_pose = self.bottle1.get_pose().p
        bottle2_pose = self.bottle2.get_pose().p
        return bottle1_pose[2] > 0.8 + self.table_z_bias and bottle2_pose[2] > 0.8 + self.table_z_bias

