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
import matplotlib.cm as cm # Import for circle colors


def visualize_results(mesh_path, position, orientation, ray_results, wsf):
    # 1. Load and Scale STL
    stl_mesh = mesh.Mesh.from_file(mesh_path)
    # stl_mesh.vectors is an array of [triangles, vertices, xyz]
    vectors = stl_mesh.vectors * wsf 

    # 2. Apply PyBullet Rotation and Translation
    # PyBullet orientation is [x, y, z, w]
    rot = R.from_quat(orientation)
    
    # Flatten to transform all vertices, then reshape back
    points = vectors.reshape(-1, 3)
    rotated_points = rot.apply(points)
    translated_points = rotated_points + np.array(position)
    final_vectors = translated_points.reshape(-1, 3, 3)

    # 3. Extract Hit Points
    hit_points = []

    for i in range(1, len(ray_results)):
        if ray_results[i][0] > -1:
            hit_points.append(ray_results[i][3])
    hit_points = np.array(hit_points)

    #print(hit_points)

    # 4. Create Figure
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 5. Plot Mesh with Edges
    # edgecolor='black' displays the wireframe/edges
    poly = mplot3d.art3d.Poly3DCollection(final_vectors, alpha=0.15)
    poly.set_facecolor('royalblue')
    poly.set_edgecolor('black') 
    poly.set_linewidth(0.3)
    ax.add_collection3d(poly)

    # 6. Plot Hit Points
    if len(hit_points) > 0:
        ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], 
                   color='red', s=10, label='Ray Hits', depthshade=False)

    # 7. Center 0,0,0 and offset Z to the bottom plane
    # Find global bounds to set axes limits
    all_dims = final_vectors.reshape(-1, 3)
    max_range = np.array([all_dims[:,0].max()-all_dims[:,0].min(), 
                          all_dims[:,1].max()-all_dims[:,1].min(), 
                          all_dims[:,2].max()-all_dims[:,2].min()]).max() / 2.0

    mid_x = (all_dims[:,0].max() + all_dims[:,0].min()) * 0.5
    mid_y = (all_dims[:,1].max() + all_dims[:,1].min()) * 0.5
    min_z = all_dims[:,2].min()

    # Set limits so object is centered in XY and sits on the Z=0 floor
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(min_z, min_z + max_range * 2) 

    mm_formatter = FuncFormatter(lambda val, pos: f'{val * 1000:.1f}')

    # Apply to all three axes
    ax.xaxis.set_major_formatter(mm_formatter)
    ax.yaxis.set_major_formatter(mm_formatter)
    ax.zaxis.set_major_formatter(mm_formatter)

    # Update labels to reflect millimeters
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    ax.set_title(f'3D Analysis: {len(hit_points)} Hits Detected')
    
    plt.legend()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "nozzle_plots")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 2. Generate unique components:
    # Timestamp: YearMonthDay_HourMinSec
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # 3. Construct the filename
    # Structure: size_sweep_mesh_compare_[pressure]_[timestamp]_[id].png
    filename = f"nozzle_hit_plot_{timestamp}.png"
    full_file_path = os.path.join(save_path, filename)
    
    # 4. Save the figure (Always call BEFORE plt.show())
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"saved nozzle plot to: {full_file_path}")

    plt.show()

def plot_ray_distribution(ray_to_coords, results, ray_spread, wsf):
    """
    Plots the X and Y coordinates of the rayTo array.
    Points are colored: Red for Hit, Blue for Miss.
    Adds concentric circles based on ray_spread radii.
    """
    ray_coords_sliced = np.array(ray_to_coords[1:])
    results_sliced = results[1:]
    
    # 2. Extract and scale coordinates to mm
    x_mm = ray_coords_sliced[:, 0] / wsf
    y_mm = ray_coords_sliced[:, 1] / wsf
    
    # 3. Determine colors for the scatter points based on hits
    colors = []
    for i in range(len(results_sliced)):
        if results_sliced[i][0] > -1:
            colors.append('red')   # Hit
        else:
            colors.append('royalblue') # Miss

    # 4. Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 5. Draw the concentric circles (Sampling Rings)
    # We create a simple color gradient for the rings
    num_circles = len(ray_spread)
    cmap = cm.get_cmap('Greys') # Using a light grey gradient
    
    for i, radius in enumerate(ray_spread):
        # Convert radius from meters back to mm
        radius_mm = radius / wsf
        
        # Adjust alpha (transparency) so inner rings are slightly bolder
        #alpha_val = 0.6 - (i * 0.4 / num_circles) if num_circles > 1 else 0.5
        
        # Create the circle patch centered at (0,0)
        circle = plt.Circle((0, 0), radius_mm, 
                            color=cmap(0.6), # A medium grey
                            fill=False, 
                            linestyle='--', 
                            linewidth=1.0, 
                            alpha=1)
        ax.add_patch(circle)

    # 6. Plot the ray points (scatter)
    # We use a higher alpha so points are clear, but smaller size 's' to not obscure rings
    ax.scatter(x_mm, y_mm, c=colors, s=8, alpha=0.7, zorder=10) # zorder ensures points are on top
    
    # 7. Formatting and Labels
    plt.title(f"Ray Sampling Pattern\n({len(ray_coords_sliced)} Rays, {num_circles} Concentric Rings)")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    
    plt.grid(True, linestyle=':', alpha=0.5) # Light grid background
    plt.axis('equal') # CRITICAL: Keeps circles circular
    
    # Set limits slightly larger than the largest circle
    if num_circles > 0:
        max_r_mm = ray_spread[-1] / wsf
        padding = max_r_mm * 0.1
        plt.xlim(-max_r_mm - padding, max_r_mm + padding)
        plt.ylim(-max_r_mm - padding, max_r_mm + padding)

    # 8. Add a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Ray End Point',
               markerfacecolor='red', markersize=9),
        #Line2D([0], [0], marker='o', color='w', label='Miss',
        #       markerfacecolor='royalblue', markersize=9),
        Line2D([0], [0], color=cmap(0.6), linestyle='--', linewidth=1.0, 
               label='Sampling Ring Borders')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "nozzle_plots")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 2. Generate unique components:
    # Timestamp: YearMonthDay_HourMinSec
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # 3. Construct the filename
    # Structure: size_sweep_mesh_compare_[pressure]_[timestamp]_[id].png
    filename = f"nozzle_hit_plot_{timestamp}.png"
    full_file_path = os.path.join(save_path, filename)
    
    # 4. Save the figure (Always call BEFORE plt.show())
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"saved nozzle plot to: {full_file_path}")

    plt.show()

def plot_solid_cylinders(ray_spread, falloff_multiplier, wsf):
    """
    Plots solid concentric cylinders with correct depth sorting.
    Outer (larger) cylinders are plotted first so smaller ones appear in front.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Data preparation
    heights = np.flip(falloff_multiplier)
    radii_mm = np.array(ray_spread) / wsf
    
    # IMPORTANT: Zip and sort by radius in DESCENDING order
    # This ensures we plot the 'Big' cylinders before the 'Small' ones
    sorted_data = sorted(zip(radii_mm, heights), key=lambda x: x[0], reverse=True)
    
    # Color map - using a sequential map to show intensity
    colors = cm.viridis(np.linspace(0, 0.9, len(sorted_data)))

    for i, (r, h) in enumerate(sorted_data):
        # Generate mesh for the wall
        theta = np.linspace(0, 2 * np.pi, 60)
        z_grid = np.linspace(0, h, 2)
        theta_grid, z_vals = np.meshgrid(theta, z_grid)
        
        x_wall = r * np.cos(theta_grid)
        y_wall = r * np.sin(theta_grid)
        
        # Plot Wall: Outer cylinders (drawn first) won't overwrite inner ones
        ax.plot_surface(x_wall, y_wall, z_vals, 
                        #color=colors[i], 
                        color='blue',
                        alpha=0.4,          # Slight transparency helps depth perception
                        edgecolor='none', 
                        shade=True,
                        antialiased=True,
                        zorder=i)           # Explicit z-order hint

        # Plot Top Cap
        r_cap = np.linspace(0, r, 10)
        theta_cap, r_vals = np.meshgrid(theta, r_cap)
        x_cap = r_vals * np.cos(theta_cap)
        y_cap = r_vals * np.sin(theta_cap)
        z_cap = np.full_like(x_cap, h)
        
        ax.plot_surface(x_cap, y_cap, z_cap, 
                        #color=colors[i], 
                        color='blue',
                        alpha=0.4, 
                        edgecolor='black', 
                        linewidth=0.2,
                        shade=True,
                        zorder=i + 0.1)

    # Aesthetics
    ax.set_title("3D Weighted Falloff (Corrected Depth)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Weight Multiplier")

    # Set a viewing angle that highlights the 'Stepped' look
    ax.view_init(elev=30, azim=220)
    
    # Standardize scaling
    ax.set_box_aspect([1, 1, 0.6]) 
    
    plt.tight_layout()
    plt.show()

def plot_cross_section(ray_spread, falloff_multiplier, wsf):
    """
    Plots a 2D cross-section silhouette with vertical lines 
    at each radius boundary defined in graph_circles.
    """
    # 1. Prepare Data
    radii_mm = np.array(ray_spread) / wsf
    heights = np.flip(falloff_multiplier)
    
    # Create the step coordinates for the outline
    x_steps = [0]
    y_steps = [heights[0]]
    
    for i in range(len(radii_mm)):
        r = radii_mm[i]
        h = heights[i]
        x_steps.append(r)
        y_steps.append(h)
        if i + 1 < len(heights):
            x_steps.append(r)
            y_steps.append(heights[i+1])

    x_right = np.array(x_steps)
    y_right = np.array(y_steps)
    x_full = np.concatenate([-x_right[::-1], x_right])
    y_full = np.concatenate([y_right[::-1], y_right])

    # 2. Create the Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Fill the profile
    ax.fill_between(x_full, 0, y_full, color='skyblue', alpha=0.2)
    ax.plot(x_full, y_full, color='navy', lw=2, label='Multiplier Profile', zorder=5)

    # 3. Draw Vertical Dividers at each radius
    # We loop through radii_mm and draw lines at both +/- positions
    for i, r in enumerate(radii_mm):
        # Draw the vertical line from y=0 to the top of the profile at that point
        h = heights[i]
        ax.vlines([r, -r], 0, h, color='red', linestyle='--', alpha=0.6, 
                  linewidth=1, label='Ring Boundary' if i == 0 else "")
        
        # Optional: Label the rings
        ax.text(r, -max(heights)*0.05, f'{r:.1f}', color='red', 
                fontsize=8, ha='center', va='top')

    # 4. Aesthetics
    ax.set_title("Nozzle Multiplier Cross-Section (with Ring Boundaries)")
    ax.set_xlabel("Radius from Nozzle Center [mm]")
    ax.set_ylabel("Velocity Multiplier")
    
    # Formatting
    ax.axvline(0, color='black', lw=1.5) # Center axis
    ax.axhline(0, color='black', lw=1)   # Ground line
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # Set limits to include labels
    ax.set_ylim(-max(heights)*0.1, max(heights) * 1.1)
    
    plt.legend(loc='upper right')
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "multiplier_plots")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 2. Generate unique components:
    # Timestamp: YearMonthDay_HourMinSec
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # 3. Construct the filename
    # Structure: size_sweep_mesh_compare_[pressure]_[timestamp]_[id].png
    filename = f"multiplier_plot_{timestamp}.png"
    full_file_path = os.path.join(save_path, filename)
    
    # 4. Save the figure (Always call BEFORE plt.show())
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"saved multiplier plot to: {full_file_path}")

    plt.show()

def plot_distribution_components(circle_number, mu, sigma, rest, average):
    """
    Plots the Gaussian probability slices with vertical separators 
    between each column for better visual distinction.
    """
    # 1. Prepare Data
    indices = np.arange(0, circle_number + 1)
    
    slices = []
    for i in range(1, circle_number + 1):
        val = (norm.cdf(i, loc=mu, scale=sigma) - 
               norm.cdf(i - 1, loc=mu, scale=sigma) + rest)
        slices.append(val)
    
    # Pad for 'post' step plotting
    slices_to_plot = slices + [slices[-1]] 

    fig, ax = plt.subplots(figsize=(12, 6))

    # 2. Plot the probability steps
    ax.step(indices, slices_to_plot, where='post', color='teal', lw=2, label='Normal distribution CDF Value', zorder=3)
    ax.fill_between(indices, slices_to_plot, step="post", alpha=0.15, color='teal', zorder=2)

    # 3. Add Vertical Separators
    # We draw lines at every integer between the bars
    for x in range(circle_number + 1):
        # Determine the height of the line based on adjacent slices
        # This keeps the lines from shooting up to the top of the graph unnecessarily
        if x == 0:
            h = slices[0]
        elif x == circle_number:
            h = slices[-1]
        else:
            h = max(slices[x-1], slices[x])
            
        ax.vlines(x, 0, h, color='teal', linestyle='-', alpha=0.3, linewidth=1, zorder=1)

    # 4. Plot the average line
    ax.axhline(y=average, color='crimson', linestyle='-', lw=2, 
               label=f'Average distribution', zorder=4)

    # 5. Aesthetics
    ax.set_title(f"Velocity shares in {circle_number} Rings")
    ax.set_xlabel("Circle Numbers")
    ax.set_ylabel("Share of total Velocity")
    
    # Ensure every index is labeled if reasonable, otherwise space them
    if circle_number <= 30:
        ax.set_xticks(range(circle_number + 1))
    
    ax.set_xlim(0, circle_number)
    ax.set_ylim(0, max(slices) * 1.1) # Add 10% headroom
    
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "nozzle_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "velocity_shares.png"), dpi=300)
    plt.show()

def calc_force(
        name_obj, 
        start_pos, 
        cw, 
        nozzle_distance, 
        nozzle_diameter, 
        nozzle_pressure,
        flow,
        ray_number,
        old_eq,
        print_results,
        graph,
        use_gui):
   
    start_time = time.time()


    #name_obj='Qf4i'
    #name_obj='1dx1h_disc'


    #use_gui = True
    wsf = 0.001  #short for world scaling factor, so 1 unit is 1 mm

    if (use_gui):
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setGravity(0, 0, -10)

    # Set camera Settings
    if (use_gui):
        p.resetDebugVisualizerCamera(
            cameraDistance=100.0*wsf,         # Increase this value to zoom further out
            cameraYaw=-35.0,             # Optional: Horizontal angle
            cameraPitch=0,          # Optional: Vertical angle
            cameraTargetPosition=[0, 0, 0] # Optional: The point in the world to look at
        )

    # Import Plane
    planeId = p.loadURDF("plane.urdf", basePosition=[0,0,0])
    p.changeVisualShape(planeId, -1, rgbaColor=[1,1,1,0])

    # Import STL
    #startPos = [-2.0*wsf,-10.0*wsf,15.0*wsf]
    #start_pos = [0.0*wsf,0.0*wsf,15.0*wsf]
    startOrientation = p.getQuaternionFromEuler([0,0,0])
    mesh_file_path = '/Users/leonardolfens/Desktop/Python_Match/pybullet/STLs/'+name_obj+'.stl' 

    # sloppy compensation for weird model
    workpiece_scale = [1.0*wsf,1.0*wsf,1.0*wsf]
    #com_offset = [40.0*wsf, 40.0*wsf, 10*wsf]
    com_offset = [0.0*wsf, 0.0*wsf, 0.0*wsf]

    # 1. Create the collision shape
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_file_path,
        meshScale=workpiece_scale,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )

    # 2. Create the visual shape 
    visual_shape_id = -1
    if (use_gui):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_file_path,
            meshScale=workpiece_scale,
            rgbaColor=[0.5, 0.5, 0.5, 1] # Optional: Set color (R, G, B, Alpha)
        )

    # 3. Combine them into a physics body
    # Mass > 0 creates a dynamic object; mass = 0 creates a static object (like the plane)
    object_mass = 1.0 
    object_id = p.createMultiBody(
        baseMass=object_mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        baseInertialFramePosition=com_offset, 
        basePosition=start_pos,
        baseOrientation=startOrientation
    )

    # reduce bouncing
    p.changeDynamics(
        bodyUniqueId=object_id, 
        linkIndex=-1,  
        restitution=0.1,
        linearDamping=0.9,   
        angularDamping=0.9
    )

    p.changeDynamics(
        bodyUniqueId=planeId, 
        linkIndex=-1,  # -1 refers to the base link of the object/plane
        restitution=0.1
    )

    #print(f"Loaded object with ID: {object_id}")

    # Define Ray origin and end points


    ray_height = 100*wsf
    ray_height = 25*wsf
    #nozzle_distance = 25*wsf
    nozzle_spread = 20.0 #[Degrees] or 23.6
    d1 = nozzle_diameter*wsf #nozzle diameter [mm]
    ray_height += d1/(2*math.tan(nozzle_spread*math.pi/180/2))
    ray_angle = 0.0
    circle_index = 0

    #ray_number = 300
    cone_diameter = ray_height*math.tan(nozzle_spread*math.pi/180/2)*2
    cone_stepsize = cone_diameter/46 #46 is good
    circle_number = round(cone_diameter/cone_stepsize)
    cone_area = (cone_diameter/2)**2*math.pi
    circle_areas = []
    circle_areas.append((cone_stepsize/2)**2*math.pi)
    donut_areas = []
    donut_areas.append((cone_stepsize/2)**2*math.pi)
    ray_spread = []
    #ray_spread.append(cone_stepsize/4)
    circle_resolutions = []

    for i in range(1,circle_number):
        circle_areas.append(((cone_stepsize+i*cone_stepsize)/2)**2*math.pi)
        donut_areas.append(circle_areas[i]-circle_areas[i-1])
        #ray_spread.append(cone_stepsize/2*i-(cone_stepsize/2))
        

    for i in range(0,circle_number):
        circle_resolutions.append(round(ray_number*donut_areas[i]/cone_area))
        ray_spread.append(cone_stepsize/2*(i+1)-cone_stepsize/4)
    
    ray_number=sum(circle_resolutions)

    #print(ray_spread)
    #print(cone_diameter)
    #print(circle_areas)
    #print(donut_areas)
    #print(ray_spread)
    #print(circle_resolutions)
    #print(sum(circle_resolutions))
    #print(ray_number)

    rayFrom = []
    rayFrom.append([0.0*wsf, 0.0*wsf, -nozzle_distance*wsf-(d1/(2*math.tan(nozzle_spread*math.pi/180/2)))])
    rayTo = []
    rayTo.append([
    rayFrom[0][0],
    rayFrom[0][1],
    rayFrom[0][2]+ray_height
        ])
    
    

    #print(rayFrom[0])
    #print(nozzle_distance)
    #print(d1)
    #print(nozzle_spread)

    rayIds = []

    rayHitColor = [1, 0, 0]
    rayMissColor = [0, 1, 0]

    replaceLines = True

    rayIds.append(p.addUserDebugLine(rayFrom[0], rayTo[0], rayMissColor))

    circle_counter = 0
    for i in range(1, ray_number+1):

        if circle_counter >= (circle_resolutions[circle_index]):
            circle_counter = 0
            ray_angle = 0.0
            circle_index += 1
        else:
            ray_angle = ray_angle + (2*math.pi)/circle_resolutions[circle_index]
        circle_counter += 1

        rayFrom.append(rayFrom[0])
        rayTo.append([
            math.sin(ray_angle) * ray_spread[circle_index],
            math.cos(ray_angle) * ray_spread[circle_index],
            rayTo[0][2]
        ])
        #print(rayTo[i])
        #print("index: "+str(circle_index)+", angle: "+str(ray_angle)+", sin: "+str(math.sin(ray_angle))+", cos: "+str(math.cos(ray_angle))+", spread. "+str(ray_spread[circle_index]))
        
        
        if (replaceLines):
            rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
        else:
            rayIds.append(-1)


        

    #print(donut_areas)
    #print(circle_resolutions)
    
    location_matrix = []
    rotation_matrix = []


    for i in range (500):
        p.stepSimulation()
        location_orientation = p.getBasePositionAndOrientation(object_id)
        location_matrix.append([
            round(location_orientation[0][0], 4), 
            round(location_orientation[0][1], 4),
            round(location_orientation[0][2], 4)
            ])
        rotation_matrix.append([
            round(location_orientation[1][0], 4), 
            round(location_orientation[1][1], 4),
            round(location_orientation[1][2], 4),
            round(location_orientation[1][3], 4),
            ])

        if (i>5 and location_matrix[i]==location_matrix[i-5] and rotation_matrix[i]==rotation_matrix[i-5]):
            #print('break!')
            object_location=location_matrix[i]
            object_rotation=rotation_matrix[i]
            break

        if (i>=499):
            object_location=location_matrix[i]
            object_rotation=rotation_matrix[i]

        if(use_gui):
            time.sleep(1./500.)



    if (not use_gui):
        timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

    numSteps = 1
    if (use_gui):
        numSteps = 1

    calctime = time.time()
    for i in range(numSteps):
    #p.stepSimulation()
    #for j in range(8):
        #results = p.rayTestBatch(rayFrom, rayTo, j + 1)
        calctime = time.time()
        results = p.rayTestBatch(rayFrom, rayTo)
        calctime = time.time()-calctime
        #print("calc time: "+str(calctime))
        #for i in range (10):
        #	p.removeAllUserDebugItems()

        if (use_gui and i<1):
            if (not replaceLines):
                p.removeAllUserDebugItems()

            for i in range(ray_number+1):
                hitObjectUid = results[i][0]

                if (hitObjectUid < 0):
                    hitPosition = [0, 0, 0]
                    p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
                else:
                    hitPosition = results[i][3]
                    p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])
                    #print(rayIds[i], results[i][3])
                
        p.stepSimulation()
        #time.sleep(1.)

    #Calculate centricity coefficient
    Cc_method = 3    #1: ray based, 2: circle based, custom distribution, 3: cirlce based, gaussian distribution
    
    falloff_multiplier = []
    average = 0.5/circle_number

    mu = circle_number
    sigma = circle_number*0.45      #3 for method 3, 40deg

    #for i in range (1, circle_number+1):   
    #    if (i>1):
    #        falloff_multiplier.append(round((norm.cdf(i, loc=mu, scale=sigma)-norm.cdf(i-1, loc=mu, scale=sigma))/average, 5))
    #    else:
    #        falloff_multiplier.append(round(norm.cdf(i, loc=mu, scale=sigma)/average,5))

    #falloff_multiplier = []
    rest = norm.cdf(0, loc=mu, scale=sigma)/circle_number
    for i in range (1, circle_number+1):  
        falloff_multiplier.append(round((norm.cdf(i, loc=mu, scale=sigma)-norm.cdf(i-1, loc=mu, scale=sigma)+rest)/average, 5))

    Cc = 1.0 #Centricity coefficioent

    if (Cc_method == 2):
        falloff_multiplier = [0.25, 0.5, 0.75, 0.75, 1, 1.25, 1.5, 2, 2, 2]
    #for i in range 

    #falloff_multiplier = np.flip(falloff_multiplier)
    #print(falloff_multiplier)
    #print(sum(falloff_multiplier))
    
    
    hits_in_circle = []

    multiplier_total = 0.0
    average2 = 0.0
    current_ray = 0
    hits = 0

    for i in range (circle_number):
        hits = 0
        average2 += falloff_multiplier[circle_number-1-i]*circle_resolutions[i]
        for j in range (circle_resolutions[i]):
            current_ray+=1
            if (results[current_ray][0] > 0):
                hits += 1
                multiplier_total+= falloff_multiplier[circle_number-1-i]
                #print(falloff_multiplier[circle_number-1-i])
                #print(current_ray)
        hits_in_circle.append(hits)

    average2 = average2/ray_number
    #print(average2) 

    Cc = round(multiplier_total/sum(hits_in_circle)/average2, 5)
    

    #print(sum(hits_in_circle))

    hit_number = sum(hits_in_circle)
    multiplier_total = 0.0
    multiplier_balance = []
    #donut_area_percentage = []

    for i in range(circle_number):
        #donut_area_percentage.append(hits_in_circle[i]/hit_number)
        multiplier_balance.append(hits_in_circle[i]/circle_resolutions[i])
        multiplier_total += multiplier_balance[i]*falloff_multiplier[circle_number-1-i]

    if (Cc_method == 2 or Cc_method == 3):
        Cc = round(multiplier_total/sum(multiplier_balance), 5)

    #print(circle_resolutions)
    #print(falloff_multiplier)
    #print(donut_area_percentage)
    #print(sum(donut_area_percentage))

    hit_number = 0
    distance_sum = 0
    calctime2 = time.time()
    for i in range (1, ray_number+1):
        if (results[i][0] > 0):
            hit_number += 1
            distance_sum += results[i][3][2] - rayFrom[i][2] - d1/(2*math.tan(nozzle_spread*math.pi/180/2))
            #print(results[i][3])
            #print(math.sqrt(results[i][3][0]**2+results[i][3][1]**2))
        #print(round(results[i][3][2], 2))
        #print(rayFrom[i][2])
        #print(results[i][3][2] - rayFrom[i][2])
    #print(distance_sum)

    hit_fraction = hit_number/ray_number

    average_distance = 0
    if not (distance_sum == 0):
        average_distance = distance_sum/hit_number/wsf
    

    
    #CALIBRATED CW
    pressures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
    new_cw = [np.float64(1.1094), 
              np.float64(1.0234), 
              np.float64(0.9767), 
              np.float64(0.9713), 
              np.float64(0.9693), 
              np.float64(0.9659), 
              np.float64(0.9695), 
              np.float64(0.9869), 
              np.float64(0.9947), 
              np.float64(1.0066), 
              np.float64(1.0286), 
              np.float64(1.0506)]
    for i in range(len(pressures)):
        if (nozzle_pressure >= pressures[i]*100000):
            cw = new_cw[i]


    #bars = nozzle_pressure/100000
    #cw = -2.054*bars**3+3.260*bars**2-1.181*bars+1.104
    #cw=1.0
    #cw = cw*0.47
    cw = cw*0.4958
    #print(cw)
    y = average_distance/1000 #distance from nozzle to object [m]
    r1 =y*math.tan(nozzle_spread*math.pi/180/2)+d1/2 #radius of projected circle
    #print(2*r1)
    Ast = r1**2*math.pi*hit_fraction #Anstroemflaeche [m^2]

    calctime2 = time.time()-calctime2
    #print("calc time 2: "+str(calctime2))
    print("calc time total: "+str(calctime+calctime2))

    roh = 1.225 #density of air [Kg/m^3]
    p1 = 101325 #ambient air pressure [Pa]
    p0 = p1 + nozzle_pressure #nozzle pressure [Pa]
    gamma = nozzle_spread 
    #cw = 1.17 #drag coefficient [-]
    kappa = 1.4 #heat capacity ratio [-]
    t0 = 293.15
    t1 = t0*(p1/p0)**((kappa-1)/kappa)
    #print(t1)
    delta_t = t0-t1
    roh0 = p0/(287.05*293.15)
    roh1 = p1/(287.05*t1)
    
    #already established further up: d1 = 0.0035 #nozzle diameter [m]
    w1 = math.sqrt(2*p0/roh0*(kappa/(kappa-1))*(1-(p1/p0)**((kappa-1)/kappa)))
    #print(w1)
    Fw  = cw*Ast*p1*(kappa/(kappa-1))*(1-(p1/p0)**((kappa-1)/kappa))*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2 #resistance force [N] 
    Fw = cw*Ast*roh*287.05*293.15*(kappa/(kappa-1))*(1-(p1/p0)**((kappa-1)/kappa))*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2
    Fw = cw*Ast*roh1*1005*293.15*(1-(p1/p0)**((kappa-1)/kappa))*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2
    Fw = cw*Ast*kappa/(kappa-1)*p1*((p0/p1)**((kappa-1)/kappa)-1)*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2
    #Fw  = cw*Ast*p1*(kappa/(kappa-1))*(1-(p1/p0)**((kappa-1)/kappa))*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2
    #Fw  = cw*Ast*p1*(kappa/(kappa-1))*(p0-p1+1500)/550000*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2 #trial and error
    #Fw  = 0.67*cw*roh/2*(2*nozzle_pressure/roh)*Ast*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2+0.015 #20deg: *0.7, +0.015
    Fwb = cw*Ast*p1*(kappa/(kappa-1))*(1-(p1/p0)**(kappa/(kappa-1)))*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2 #Bansman method used in Matlab, still contains mistake

    area_comp = (d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2

    if(old_eq):
        #Fw = cw*roh/2*((flow*1000/(2*math.pi))**2)*Ast*(d1/(d1+2*y*math.sin(gamma*math.pi/180/2)))**2
        #w2 = math.sqrt((flow*1000/(4*math.pi))**2+(2*nozzle_pressure/roh)*(1-(nozzle_pressure/(2*kappa*p1))))
        #w2 = math.sqrt((flow*1000/(4*math.pi))**2+2*kappa/(kappa-1)*((p0/roh0)-(p1/roh1)))
        #print(p0/roh0, p1/roh)
        #Fw = cw*roh1/2*(w2**2)*Ast*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2
        Fw  = cw*Ast*p1*(kappa/(kappa-1))*(1-(p1/p0)**((kappa-1)/kappa))*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2 #resistance force [N] 
        #Cc = 1.0

    tan = math.tan(nozzle_spread*math.pi/180/2)
    
    #finding angle problem
    #print('angle: '+str(nozzle_spread))  
    #print('hit fraction: '+str(hit_fraction))  
    #print('Ast: '+str(Ast))
    #print('area comp term: '+str(area_comp))
    #print('Ast* area comp term: '+str(Ast*area_comp))
    #print('Ast* area comp term 2: '+str((y*math.tan(nozzle_spread*math.pi/180/2)+d1/2)**2*math.pi*(d1/(d1+2*y*math.tan(gamma*math.pi/180/2)))**2))
    #print('nozzle area: '+str((d1/2)**2*math.pi))
    #print('sin: '+str(math.tan(gamma*math.pi/180/2)))
    #print('tan: '+str(math.tan(nozzle_spread*math.pi/180/2)))
    
    

    if (print_results):
        print('Object name: '+name_obj)
        print('Object location: '+str(object_location))
        print('Object rotation: '+str(object_rotation))
        print('average distance: '+str(round(average_distance, 2))+' mm')
        print('Number of Hits: '+str(hit_number)+' out of '+str(ray_number)+' ('+str(round(hit_fraction*100))+'%)')
        print('Centricity coefficient: '+str(Cc))
        print('Anstroemflaeche: '+str(Ast*1000000)+' mm^2')
        print('Resistance force: '+str(Fw)+' N (with Nozzle pressure '+str(p0)+' Pa)')

        workpiece_data = open('/Users/leonardolfens/Desktop/Python_Match/pybullet/Output_txt/'+name_obj+'_simulated_data.txt','w')
        workpiece_data.writelines("-------------------------------------------")
        workpiece_data.writelines('\n'+'Simulated_Data_'+name_obj+'\n')
        workpiece_data.writelines("-------------------------------------------"+'\n')
        workpiece_data.writelines('Object location: '+str(object_location)+'\n')
        workpiece_data.writelines('Object rotation: '+str(object_rotation)+'\n')
        workpiece_data.writelines('average distance: '+str(round(average_distance, 2))+' mm\n')
        workpiece_data.writelines('Number of Hits: '+str(hit_number)+' out of '+str(ray_number)+' ('+str(round(hit_fraction*100))+'%)\n')
        workpiece_data.writelines('Resistance force: '+str(Fw)+' N (with Nozzle pressure '+str(p0)+' Pa)\n')
        workpiece_data.close()

    

    if (not use_gui):
        p.stopStateLogging(timingLog)
        p.disconnect()
    else:
        while (p.isConnected()):
            p.stepSimulation()
            time.sleep(1./240.)



    if (graph):
        # Your existing 3D plot
        visualize_results(mesh_file_path, object_location, object_rotation, results, wsf)
        
        graph_circles = []
        for i in range(len(ray_spread)):
            graph_circles.append(ray_spread[i]+cone_stepsize/4)

        # NEW: Your 2D Ray pattern plot
        # Note: rayTo and results are both lists of length ray_number + 1
        plot_ray_distribution(rayTo, results, graph_circles, wsf)

        #plot_solid_cylinders(graph_circles, falloff_multiplier, wsf)

        plot_cross_section(graph_circles, falloff_multiplier, wsf)

    end_time = time.time()
    runtime = end_time-start_time

    #plot_distribution_components(circle_number, mu, sigma, rest, average)

    if (print_results):
        print('Runtime: '+str(runtime*1000)+' ms')

    #Cc = 1
    return Fw, Ast, hit_number, Cc
    

#rays = 2000
#calc_force('0_5dx1h_disc', [0,0,0], 1.17, 26, 4, 60000, 300, False, True, True, False)
#Fw, Ast, hit_number, Cc = calc_force('0_5dx1h_disc', [0,0,0], 1.17, 25, 4, 40, 0, rays, False, False, False, False)
#print((Ast*1000000/19.63-1)*100, Ast*1000000, hit_number)
#Fw, Ast, hit_number, Cc = calc_force('0_75dx1h_disc', [0,0,0], 1.17, 25, 4, 40, 0, rays, False, False, False, False)
#print((Ast*1000000/44.18-1)*100, Ast*1000000, hit_number)
#Fw, Ast, hit_number, Cc = calc_force('1dx1h_disc', [0,0,0], 1.17, 25, 4, 40, 0, rays, False, False, False, False)
#print((Ast*1000000/78.54-1)*100, Ast*1000000, hit_number)
#Fw, Ast, hit_number, Cc = calc_force('1_5dx1h_disc', [0,0,0], 1.17, 25, 4, 40, 0, rays, False, False, False, False)
#print(Ast*1000000, hit_number)
calc_force('4dx1h_disc', [0,0,0], 1.17, 25, 4, 40, 0, 2000, False, False, False, False)#5mm_rectangle

