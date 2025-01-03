import os
# Get the directory of this script (my_current.py)
current_dir = os.path.dirname(os.path.abspath(__file__))


from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg ,CollisionPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg

Nailing_aimingram = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_aimingarm",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_aimingarm.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.162),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_bar",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[0.71, 0, 0, 0.71]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_bar.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.114),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_gear = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_gear",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_gear.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.084),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_insertionhandle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_insertionhandle",
        # Error: W
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[0.71, 0, 0, 0.71]), #need adjustment
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_insertionhandle.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.323),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_nail = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_nail",
        # Change the orientation or position to keep object in camera frame for AnyGrasp to generation Grasp Pose
        # Error Pose: pos=[0.5, 0.0, 0.0], rot=[1, 0, 0, 0] 
        # Error when the grasping position is too close to the table: The friction force interupts the grasping process 
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.1, 0.0], rot=[1, 0, 0, 0]), 
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_nail.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.172),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_roller = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_roller",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[0.71, 0, 0.71, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_roller.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.023),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_screw = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_screw",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_screw.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.016),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_spacer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_spacer",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_spacer.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.024),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )

Nailing_stick = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nailing_stick",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.1, 0.0], rot=[0, 1, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(current_dir, "Nailing_stick.usd"),
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.205),
            collision_props=CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                min_torsional_patch_radius=0.008,
                rest_offset=0,
                torsional_patch_radius=0.1,),
            ),
        )
