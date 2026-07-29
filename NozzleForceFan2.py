import pybullet as p
import time
import math
import pybullet_data
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
from stl import mesh
from scipy.spatial.transform import Rotation as R
import os
import datetime
from datetime import datetime
import matplotlib.cm as cm 

def visualize_results(mesh_path, position, orientation, ray_results, wsf):
    stl_mesh = mesh.Mesh.from_file(mesh_path)
    vectors = stl_mesh.vectors * wsf 

    rot = R.from_quat(orientation)
    points = vectors.reshape(-1, 3)
    rotated_points = rot.apply(points)
    translated_points = rotated_points + np.array(position)
    final_vectors = translated_points.reshape(-1, 3, 3)

    hit_points = []
    # Loop over all batch results (skipping structural padding if any, checking hit status)
    for i in range(len(ray_results)):
        if ray_results[i][0] > -1:
            hit_points.append(ray_results[i][3])
    hit_points = np.array(hit_points)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    poly = mplot3d.art3d.Poly3DCollection(final_vectors, alpha=0.15)
    poly.set_facecolor('royalblue')
    poly.set_edgecolor('black') 
    poly.set_linewidth(0.3)
    ax.add_collection3d(poly)

    if len(hit_points) > 0:
        ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], 
                   color='red', s=10, label='Ray Hits', depthshade=False)

    all_dims = final_vectors.reshape(-1, 3)
    max_range = np.array([all_dims[:,0].max()-all_dims[:,0].min(), 
                          all_dims[:,1].max()-all_dims[:,1].min(), 
                          all_dims[:,2].max()-all_dims[:,2].min()]).max() / 2.0

    mid_x = (all_dims[:,0].max() + all_dims[:,0].min()) * 0.5
    mid_y = (all_dims[:,1].max() + all_dims[:,1].min()) * 0.5
    min_z = all_dims[:,2].min()

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(min_z, min_z + max_range * 2) 

    mm_formatter = FuncFormatter(lambda val, pos: f'{val * 1000:.1f}')
    ax.xaxis.set_major_formatter(mm_formatter)
    ax.yaxis.set_major_formatter(mm_formatter)
    ax.zaxis.set_major_formatter(mm_formatter)

    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')
    ax.set_title(f'3D Analysis: {len(hit_points)} Hits Detected Across Array')
    plt.legend()
    plt.show()

def calc_force_fan(
        name_obj, 
        start_pos, 
        cw, 
        nozzle_distance, 
        nozzle_diameter, 
        nozzle_pressure,
        ray_number,
        print_results,
        graph,
        use_gui,
        num_nozzles=8,               
        nozzle_offset=[2.8, 0, 0]):  
   
    start_time = time.time()
    wsf = 0.001  # World scaling factor (1 unit = 1 mm)

    if (use_gui):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -10)

    if (use_gui):
        p.resetDebugVisualizerCamera(
            cameraDistance=100.0*wsf,         
            cameraYaw=-35.0,             
            cameraPitch=0,          
            cameraTargetPosition=[0, 0, 0] 
        )

    planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0])
    p.changeVisualShape(planeId, -1, rgbaColor=[1,1,1,0])

    startOrientation = p.getQuaternionFromEuler([0,0,0])
    mesh_file_path = '/Users/leonardolfens/Desktop/Python_Match/pybullet/STLs/'+name_obj+'.stl' 

    workpiece_scale = [1.0*wsf,1.0*wsf,1.0*wsf]
    com_offset = [0.0*wsf, 0.0*wsf, 0.0*wsf]

    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_file_path,
        meshScale=workpiece_scale,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    visual_shape_id = -1
    if (use_gui):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_file_path,
            meshScale=workpiece_scale,
            rgbaColor=[0.5, 0.5, 0.5, 1] 
        )

    object_mass = 1.0 
    object_id = p.createMultiBody(
        baseMass=object_mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        baseInertialFramePosition=com_offset, 
        basePosition=start_pos,
        baseOrientation=startOrientation
    )

    p.changeDynamics(bodyUniqueId=object_id, linkIndex=-1, restitution=0.1, linearDamping=0.9, angularDamping=0.9)
    p.changeDynamics(bodyUniqueId=planeId, linkIndex=-1, restitution=0.1)

    # Base configuration parameters for one nozzle profile
    nozzle_spread = 20.0 
    d1 = nozzle_diameter*wsf 
    
    # Calculate geometric distribution variables for a single nozzle pattern
    base_ray_height = 50*wsf + d1/(2*math.tan(nozzle_spread*math.pi/180/2))
    cone_diameter = base_ray_height*math.tan(nozzle_spread*math.pi/180/2)*2
    cone_stepsize = cone_diameter/15
    circle_number = round(cone_diameter/cone_stepsize)
    cone_area = (cone_diameter/2)**2*math.pi
    
    circle_areas = [(cone_stepsize/2)**2*math.pi]
    donut_areas = [(cone_stepsize/2)**2*math.pi]
    ray_spread = []
    circle_resolutions = []

    for i in range(1, circle_number):
        circle_areas.append(((cone_stepsize+i*cone_stepsize)/2)**2*math.pi)
        donut_areas.append(circle_areas[i]-circle_areas[i-1])      

    for i in range(0, circle_number):
        circle_resolutions.append(round(ray_number*donut_areas[i]/cone_area))
        ray_spread.append(cone_stepsize/2*(i+1)-cone_stepsize/4)
    
    rays_per_nozzle = sum(circle_resolutions)

    # Initialize master lists containing data across all generated nozzles
    rayFrom = []
    rayTo = []
    rayIds = []
    rayMissColor = [0, 1, 0]
    rayHitColor = [1, 0, 0]

    offset_scaled = np.array(nozzle_offset) * wsf
    start_nozzle_index_offset = -(num_nozzles - 1) / 2.0

    # Generate full batched ray configurations
    for n in range(num_nozzles):
        n_offset = (start_nozzle_index_offset + n) * offset_scaled
        
        local_ray_from_base = [n_offset[0], n_offset[1], -nozzle_distance*wsf-(d1/(2*math.tan(nozzle_spread*math.pi/180/2))) + n_offset[2]]
        local_ray_to_base = [local_ray_from_base[0], local_ray_from_base[1], local_ray_from_base[2] + base_ray_height]
        
        rayFrom.append(local_ray_from_base)
        rayTo.append(local_ray_to_base)
        if use_gui:
            rayIds.append(p.addUserDebugLine(local_ray_from_base, local_ray_to_base, rayMissColor))
        else:
            rayIds.append(-1)

        circle_index = 0
        circle_counter = 0
        ray_angle = 0.0
        
        for i in range(rays_per_nozzle):
            if circle_counter >= (circle_resolutions[circle_index]):
                circle_counter = 0
                ray_angle = 0.0
                circle_index += 1
            else:
                ray_angle = ray_angle + (2*math.pi)/circle_resolutions[circle_index]
            circle_counter += 1

            rayFrom.append(local_ray_from_base)
            rayTo.append([
                local_ray_to_base[0] + math.sin(ray_angle) * ray_spread[circle_index],
                local_ray_to_base[1] + math.cos(ray_angle) * ray_spread[circle_index],
                local_ray_to_base[2]
            ])
            
            if use_gui:
                rayIds.append(p.addUserDebugLine(rayFrom[-1], rayTo[-1], rayMissColor))
            else:
                rayIds.append(-1)

    # Run physics settling steps
    location_matrix = []
    rotation_matrix = []

    for i in range (500):
        p.stepSimulation()
        location_orientation = p.getBasePositionAndOrientation(object_id)
        location_matrix.append([round(x, 4) for x in location_orientation[0]])
        rotation_matrix.append([round(x, 4) for x in location_orientation[1]])

        if (i>5 and location_matrix[i]==location_matrix[i-5] and rotation_matrix[i]==rotation_matrix[i-5]):
            object_location=location_matrix[i]
            object_rotation=rotation_matrix[i]
            break
        if (i>=499):
            object_location=location_matrix[i]
            object_rotation=rotation_matrix[i]
        if(use_gui):
            time.sleep(1./500.)

    # Fire all rays in one batch
    results = p.rayTestBatch(rayFrom, rayTo)

    if use_gui:
        for idx in range(len(rayFrom)):
            hitObjectUid = results[idx][0]
            if hitObjectUid < 0:
                p.addUserDebugLine(rayFrom[idx], rayTo[idx], rayMissColor, replaceItemUniqueId=rayIds[idx])
            else:
                p.addUserDebugLine(rayFrom[idx], results[idx][3], rayHitColor, replaceItemUniqueId=rayIds[idx])
        p.stepSimulation()

    # Dynamic Array accumulation outputs
    total_force = 0.0
    total_astroem_area = 0.0
    total_hits_all_nozzles = 0
    individual_nozzle_data = []

    total_rays_per_nozzle_block = rays_per_nozzle + 1  

    # Process each nozzle individually from sliced batch results
    for n in range(num_nozzles):
        start_idx = n * total_rays_per_nozzle_block
        end_idx = start_idx + total_rays_per_nozzle_block
        nozzle_results = results[start_idx:end_idx]
        nozzle_ray_from = rayFrom[start_idx:end_idx]

        hits_in_circle = []
        current_ray = 0

        # Gather hit counts for each concentric ring layer
        for i in range(circle_number):
            hits = 0
            for j in range(circle_resolutions[i]):
                current_ray += 1
                if nozzle_results[current_ray][0] > 0:
                    hits += 1
            hits_in_circle.append(hits)

        hit_number = sum(hits_in_circle)
        
        # --- NEW INTEGRATED CENTRICITY COEFFICIENT (Cc) LOGIC ---
        falloff_multiplier = []
        average = 0.5 / circle_number

        mu = circle_number
        sigma = circle_number * 0.45      

        rest = norm.cdf(0, loc=mu, scale=sigma) / circle_number
        for i in range(1, circle_number + 1):  
            falloff_multiplier.append(round((norm.cdf(i, loc=mu, scale=sigma) - norm.cdf(i - 1, loc=mu, scale=sigma) + rest) / average, 5))

        Cc = 1.0 
        multiplier_total = 0.0
        multiplier_balance = []

        for i in range(circle_number):
            # Safe fraction handling just in case resolution configuration shifts
            res = circle_resolutions[i] if circle_resolutions[i] > 0 else 1
            multiplier_balance.append(hits_in_circle[i] / res)
            multiplier_total += multiplier_balance[i] * falloff_multiplier[circle_number - 1 - i]

        # Prevent ZeroDivisionError if there are absolutely zero hits on this nozzle
        if sum(multiplier_balance) > 0:
            Cc = round(multiplier_total / sum(multiplier_balance), 5)
        else:
            Cc = 1.0
        # -------------------------------------------------------

        # Calculate heights and distances
        distance_sum = 0
        for i in range(1, total_rays_per_nozzle_block):
            if nozzle_results[i][0] > 0:
                distance_sum += nozzle_results[i][3][2] - nozzle_ray_from[i][2] - d1/(2*math.tan(nozzle_spread*math.pi/180/2))

        hit_fraction = hit_number / rays_per_nozzle
        average_distance = (distance_sum / hit_number / wsf) if hit_number > 0 else 0.0
        
        y = average_distance / 1000 
        r1 = y * math.tan(nozzle_spread * math.pi / 180 / 2) + d1 / 2 
        Ast = r1**2 * math.pi * hit_fraction 

        # Ideal Fluid/Aerodynamic Expansion Calculations
        p1 = 101325 
        p0 = p1 + nozzle_pressure 
        k = 1.4 
        gamma = nozzle_spread
        C_dis = 0.8 

        # Choked flow calculation
        Fw = 0.49*Ast * (p0 * (2/(k+1))**(k/(k-1)) * (1 + C_dis * k) - p1) * (d1 / (d1 + 2 * y * math.tan(gamma * math.pi / 180 / 2)))**2 
        
        Fw_scaled = Fw * Cc 

        # Accumulate results
        total_force += Fw_scaled
        total_astroem_area += Ast
        total_hits_all_nozzles += hit_number
        
        individual_nozzle_data.append({
            "nozzle_index": n,
            "hits": hit_number,
            "Cc": Cc,
            "force": Fw_scaled,
            "distance": average_distance
        })

    # Logging and Printing outputs 
    if print_results:
        print('==================================================')
        print(f'FAN NOZZLE SIMULATION RESULTS: {num_nozzles} Nozzles')
        print('==================================================')
        print('Object name: ' + name_obj)
        print('Object location: ' + str(object_location))
        print('Total Hits Across Array: ' + str(total_hits_all_nozzles) + ' / ' + str(rays_per_nozzle * num_nozzles))
        print(f'Total Accumulated Force: {total_force:.6f} N')
        print(f'Total Combined Aerodynamic Area (Ast): {total_astroem_area * 1000000:.2f} mm^2')
        print('--------------------------------------------------')
        for data in individual_nozzle_data:
            print(f" Nozzle #{data['nozzle_index']}: Hits: {data['hits']} | Cc: {data['Cc']:.4f} | Force: {data['force']:.5f} N | Dist: {data['distance']:.1f} mm")

    if use_gui:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1./240.)
    else:
        p.disconnect()

    if graph:
        visualize_results(mesh_file_path, object_location, object_rotation, results, wsf)

    return total_force, total_astroem_area, total_hits_all_nozzles, individual_nozzle_data

#calc_force_fan('2dx1h_disc', [0,0,0], 1, 30, 0.8, 200000, 500, True, True, False)