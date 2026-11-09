"""Synchronize all outer-sphere bundles and projection presets after generation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'),str(PROJECT.parent/'collision_shpere_generation')]


def sync(projection):
    from robot_workspace_dexterous.joint_contacts import merge_contact_ignores
    from collision_shpere_generation.kinematics import RobotKinematics
    records = []
    (projection/'configs').mkdir(exist_ok=True)
    camera = yaml.safe_load((projection/'config.example.yaml').read_text('utf-8'))['camera']
    for preset in sorted((PROJECT/'configs').glob('*.yaml')):
        name = preset.stem
        model = PROJECT/'models'/name
        outer = model/'collision_spheres_outer.yaml'
        if not outer.exists():
            raise FileNotFoundError(f'Generate outer spheres first: {outer}')
        config = yaml.safe_load(preset.read_text('utf-8'))
        config['robot']['collision_spheres'] = f'../models/{name}/collision_spheres_outer.yaml'
        # The exported URDF has no mesh geometry; sphere runs are fully portable.
        config['robot'].pop('collision_meshes',None)
        config['solver']['collision_backend'] = 'spheres'
        config['solver']['backend'] = 'torch'
        preset.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        target = projection/'models'/name
        target.mkdir(parents=True,exist_ok=True)
        for filename in ('robot.urdf','collision_spheres_outer.yaml','collision_spheres_outer.json','self_collision_ignore.yaml'):
            shutil.copyfile(model/filename,target/filename)
        ignores = yaml.safe_load((model/'self_collision_ignore.yaml').read_text('utf-8'))['self_collision_ignore']
        effective = merge_contact_ignores(str(model/'robot.urdf'),ignores)
        library = {'robot_cfg':{'kinematics':{
            'format_version':2.0,'urdf_path':'robot.urdf','asset_root_path':'.',
            'base_link':config['robot']['base_link'],'tool_frames':config['robot']['ee_links'],
            'collision_spheres':'collision_spheres_outer.yaml',
            'self_collision_ignore':effective,'collision_sphere_buffer':0.0}}}
        (target/'kinematics.yaml').write_text(yaml.safe_dump(library,sort_keys=False),encoding='utf-8')
        kinematics = RobotKinematics(model/'robot.urdf',config['robot']['base_link'])
        static = [link for link in yaml.safe_load(outer.read_text('utf-8'))['collision_spheres'] if kinematics.is_static(link)]
        grid,solver = config['grid'],config['solver']
        pconfig = {'robot':{'urdf':f'../models/{name}/robot.urdf',
            'library_config':f'../models/{name}/kinematics.yaml',
            'base_link':config['robot']['base_link'],'ee_links':config['robot']['ee_links']},
            'collision':{'self_collision':True,'static_body_links':static,'target_clearance':0.0},
            'plane':{'height':.3,'height_min':grid['z_min'],'height_max':grid['z_max'],
                     'height_step':grid['z_step'],'precompute_heights':True,
                     'precompute_height_step':grid['z_step'],'tolerance':grid['z_step']/2},
            'sampling':{'method':'task_ik','backend':'torch','x_range':grid['x_range'],
                        'y_range':grid['y_range'],'resolution':grid['resolution'],
                        'orientation_wxyz':None,'ik_seeds':solver['ik_seeds'],
                        'ik_iterations':solver.get('ik_iterations',120),'batch_size':solver['batch_size'],
                        'position_tolerance':solver['position_tolerance'],
                        'orientation_tolerance':solver['orientation_tolerance']},
            'camera':deepcopy(camera)}
        from robot_workspace_dexterous.sampling import generate_uniform_quaternions
        pconfig['sampling']['orientation_samples_wxyz'] = generate_uniform_quaternions(config['orientations']['count']).tolist()
        (projection/'configs'/f'{name}.yaml').write_text(yaml.safe_dump(pconfig,sort_keys=False),encoding='utf-8')
        records.append({'model':name,'base_link':config['robot']['base_link'],
                        'ee_links':config['robot']['ee_links'],'collision_mode':'external',
                        'resolution_m':grid['resolution'],'orientations':config['orientations']['count']})
    # Replace the default legacy handwritten sphere set with the reviewed library.
    for preset in (PROJECT/'configs'/'cpu').glob('*.yaml'):
        config = yaml.safe_load(preset.read_text('utf-8'))
        name = preset.stem
        config['robot']['collision_spheres'] = f'../../models/{name}/collision_spheres_outer.yaml'
        config['robot'].pop('collision_meshes',None)
        config['solver']['collision_backend'] = 'spheres'
        preset.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    default = projection/'models'/'dual_v2_2_left_hand_right_gripper_half'
    shutil.copyfile(default/'robot.urdf',projection/'dual_arm_half.urdf')
    library = yaml.safe_load((default/'kinematics.yaml').read_text('utf-8'))
    library['robot_cfg']['kinematics']['urdf_path'] = 'dual_arm_half.urdf'
    library['robot_cfg']['kinematics']['collision_spheres'] = 'models/dual_v2_2_left_hand_right_gripper_half/collision_spheres_outer.yaml'
    (projection/'dual_arm.yaml').write_text(yaml.safe_dump(library,sort_keys=False),encoding='utf-8')
    for project in (PROJECT,projection):
        (project/'models'/'outer_models_inventory.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'Synchronized {len(records)} models, including P2AB, P2AB-2 and P2AB-3',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection',type=Path,default=PROJECT.parent/'robot_workspace_projection')
    sync(parser.parse_args().projection)
