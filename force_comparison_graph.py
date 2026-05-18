import csv
import statistics
import numpy as np
import math
from scipy import stats
from nozzleForce import calc_force
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os

def extract_averages_force(file_path):
    averages = []
    data_points = []
    
    # Read the file
    with open(file_path, mode='r', encoding='utf-8') as f:
        # Using delimiter ';' based on your example
        reader = list(csv.reader(f, delimiter=';'))
        
    row_count = len(reader)
    i = 0
    
    while i < row_count:
        try:
            # Convert European decimal (,) to Python float (.)
            value = float(reader[i][1].replace(',', '.'))
        except (IndexError, ValueError):
            i += 1
            continue

        # Trigger: Value is greater than 0
        if value > 0.01:
            current_batch = []
            
            # 1. Collect all consecutive values > 0
            while i < row_count:
                try:
                    val = float(reader[i][1].replace(',', '.'))
                    if val <= 0.01:
                        break
                    current_batch.append(val)
                    i += 1
                except (IndexError, ValueError):
                    break
            
            # 2. Calculate average and append to array
            if current_batch:
                avg = sum(current_batch) / len(current_batch)
                averages.append(avg)
            
            # 3. Skip the next 600 rows
            i += 600
        else:
            # Move to next row if no trigger
            i += 1
            
    return averages

def extract_flow_data(file_path):
    flow_batches = []
    current_batch = []
    pressure_batches = []
    current_batch_pressure = []
    
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            # Using delimiter=';' to match your file format
            reader = csv.DictReader(file, delimiter=';')
            
            for row in reader:
                # Convert the Flow value to a float and add to current batch
                flow_value = float(row['Flow'])
                tank_pressure_value = float(row['pressure before Valve'])
                current_batch.append(flow_value)
                current_batch_pressure.append(tank_pressure_value)
                
                # Once we hit 100 entries, append the batch and reset
                if len(current_batch) == 100:
                    flow_batches.append(sum(current_batch)/1200)
                    current_batch = []

                if len(current_batch_pressure) == 1000:
                    pressure_batches.append(sum(current_batch_pressure)/1000)
                    current_batch_pressure = []
            

                
    except FileNotFoundError:
        print("Error: The file was not found.")
    except KeyError:
        print("Error: Column 'Flow' not found. Check your file headers.")
        
    return flow_batches, pressure_batches

def extract_averages_force_batch(file_paths):
    averages = []
    for i in range(0, len(file_paths)):
        averages.append(extract_averages_force(file_paths[i]))

    return averages

def calculate_binned_stats(data_list, bin_size=10, confidence=0.95):
    """
    Groups data into chunks, calculating the mean 
    and the 95% Confidence Interval (margin of error) for each group.
    """
    binned_averages = []
    binned_cis = [] # This now stores the margin of error
    
    for i in range(0, len(data_list), bin_size):
        chunk = data_list[i : i + bin_size]
        n = len(chunk)
        
        if n >= 2:
            avg = sum(chunk) / n
            std_err = stats.sem(chunk) # Standard Error = std / sqrt(n)
            
            # Calculate the confidence interval margin
            # h = t * (s / sqrt(n))
            h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
            
            binned_averages.append(avg)
            binned_cis.append(h)
        elif n == 1:
            binned_averages.append(chunk[0])
            binned_cis.append(0.0) 
            
    return binned_averages, binned_cis

def calculate_binned_stats_batch(data_list, bin_size=10, confidence=0.95):
    averages = []
    cis = []
    for i in range(0, len(data_list)):
        average, ci = calculate_binned_stats(data_list[i], bin_size, confidence)
        averages.append(average)
        cis.append(ci)

    return averages, cis
    
def plot_results(pressures, experimental_forces, force_cis, simulated_forces, diameter, distance):
    plt.figure(figsize=(10, 6))
    
    # 1. Perform Polynomial Fit (Degree 2 for physical force curves)
    z = np.polyfit(pressures, experimental_forces, 2)
    p = np.poly1d(z)

    # 2. Calculate Theoretical Force: F = 2 * A * P
    # A = pi * r^2 (radius in meters)
    radius_m = (4 / 2) / 1000 
    area_m2 = math.pi * (radius_m**2)
    # P in Pascals = P_bar * 100,000
    theoretical_forces = [0.5 * 2 * area_m2 * (p * 100000) for p in pressures]
    
    # Calculate R-squared
    y_fit = p(pressures)
    residuals = experimental_forces - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((experimental_forces - np.mean(experimental_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 2. Plot Experimental Data with Error Bars (No line connecting dots)
    plt.errorbar(
        pressures, experimental_forces, yerr=force_cis, 
        fmt='o', capsize=5, label='Experimental Data (Avg ± 95% CI)', color='blue', markersize=6
    )
    
    # 3. Plot the trendline
    plt.plot(pressures, p(pressures), "b--", alpha=0.6, 
             label=f'Trendline ($R^2 = {r_squared:.4f}$)')
    
    # 4. Plot Simulated Data
    plt.plot(pressures, simulated_forces, marker='s', linestyle='--', 
             label='Simulated Force', color='red', alpha=0.8)
    
    
    
    plt.title(f'Experimental Fit vs. Simulation (D: {diameter}mm, Dist: {distance}mm)')
    plt.xlabel('Nozzle Pressure (Bar)')
    plt.ylabel('Resulting Force (N)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_results2(pressures, experimental_forces, force_cis, simulated_forces, diameter, distance):
    plt.figure(figsize=(10, 6))
    
    # 1. Convert to numpy arrays for calculation
    pressures = np.array(pressures)
    exp_forces = np.array(experimental_forces)
    sim_forces = np.array(simulated_forces)
    
    # 2. Calculate R-squared (Simulated fit to Experimental)
    # Residuals = Experimental Observed - Simulated Predicted
    residuals = exp_forces - sim_forces
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((exp_forces - np.mean(exp_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # 3. Optional: Theoretical Force Calculation (Keeping for reference)
    radius_m = (diameter / 2) / 1000  # Updated to use the 'diameter' parameter
    area_m2 = math.pi * (radius_m**2)
    # P in Pascals = P_bar * 100,000
    theoretical_forces = [0.5 * 2 * area_m2 * (p * 100000) for p in pressures]
    
    # 4. Plot Experimental Data (fmt='o-' connects the experimental points)
    plt.errorbar(
        pressures, exp_forces, yerr=force_cis, 
        fmt='o-', capsize=5, label=f'Experimental Data (at {diameter}mm)', 
        color='blue', markersize=8, linewidth=1.5
    )
    
    # 5. Plot Simulated Data
    # The R-squared value is now integrated into this label
    plt.plot(
        pressures, sim_forces, marker='s', linestyle='--', 
        label=f'Simulated Force ($R^2 = {r_squared:.4f}$)', 
        color='red', alpha=0.8
    )
    
    # --- AXIS CONSTRAINTS ---
    plt.xlim(left=0, right=max(pressures) * 1.1)
    max_y = max(max(exp_forces + np.array(force_cis)), max(sim_forces))
    plt.ylim(bottom=0, top=max_y * 1.1)

    # Formatting
    plt.title(f'Experimental vs. Simulation: Pressure Sweep (D: {diameter}mm, Dist: {distance}mm)')
    plt.xlabel('Nozzle Pressure (Bar)')
    plt.ylabel('Resulting Force (N)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_results_sim_compare(pressures, experimental_forces, force_cis, simulated_forces, simulated_forces2, diameter, distance):
    plt.figure(figsize=(10, 6))
    
    cw = []
    old_fw = []
    for i in range(len(pressures)):
        cw.append(experimental_forces[i]/simulated_forces[i])
        old_fw.append(1.1*math.pi*0.004**2/4*101325*1.4/(1.4-1)*(1-(101325/(pressures[i]*100000+101325))**((1.4-1)/1.4)))

    #print(cw)
    a, b, c, d = np.polyfit(pressures, cw, 3)
    
    #print(a, b, c, d)

    # 1. Perform Polynomial Fit (Degree 2 for physical force curves)
    z = np.polyfit(pressures, experimental_forces, 2)
    p = np.poly1d(z)
    

    
    # Calculate R-squared
    y_fit = p(pressures)
    residuals = experimental_forces - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((experimental_forces - np.mean(experimental_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 2. Plot Experimental Data with Error Bars (No line connecting dots)
    plt.errorbar(
        pressures, experimental_forces, linestyle='-', yerr=force_cis, 
        fmt='o', capsize=5, label='Experimental Data (Avg ± 95% CI)', color='blue', markersize=6
    )
    
    # 3. Plot the trendline
    #plt.plot(pressures, p(pressures), "b--", alpha=0.6, 
    #         label=f'Trendline ($R^2 = {r_squared:.4f}$)')
    
    # 4. Plot Simulated Data
    plt.plot(pressures, simulated_forces, marker='s', linestyle='--', 
             label='New Fd Equation', color='green', alpha=0.8)
    
    plt.plot(pressures, simulated_forces2, marker='s', linestyle='--', 
             label='Old Fd Equation', color='red', alpha=0.8)
    
    #plt.plot(pressures, old_fw, marker='s', linestyle='--', 
    #         label='old Fw', color='purple', alpha=0.8)
    
    #theoretical_forces = [p**1.2+0.04 for p in pressures]
    cw = []
    for i in range(len(pressures)):
        cw.append(sim_forces2[i]*((pressures[i]**1.27+0.02)/(1.1*pressures[i])))
    # 5. Theoretical "Ideal" Force (F = 2AP)
    #plt.plot(pressures, theoretical_forces, marker='s', linestyle='--', 
    #         label='stuff', color='black', alpha=0.8)
    
    plt.title(f'Experimental Fit vs. Simulation (D: {diameter}mm, Dist: {distance}mm)')
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Resulting Force [N]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "sim_compare1.png"), dpi=300)
    plt.show()

def plot_size_sweep(diameters, batch_experimental_averages, batch_experimental_cis, simulated_forces, pressure, distance):
    """
    Plots the force vs. diameter for a specific fixed pressure.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Convert lists to numpy arrays for easier manipulation if they aren't already
    diameters = np.array(diameters)
    exp_forces = np.array(batch_experimental_averages)
    exp_cis = np.array(batch_experimental_cis)
    sim_forces = np.array(simulated_forces)

    # 2. Trendline for Experimental Data (Polynomial fit)
    # Using degree 2 or 1 depending on the expected physical behavior of your discs
    z = np.polyfit(diameters, exp_forces, 2)
    p = np.poly1d(z)
    
    # Calculate R-squared
    y_fit = p(diameters)
    residuals = exp_forces - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((exp_forces - np.mean(exp_forces))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # 3. Plot Experimental Data
    plt.errorbar(
        diameters, exp_forces, yerr=exp_cis, 
        fmt='o', capsize=5, label=f'Experimental (at {pressure} Bar)', 
        color='blue', markersize=8
    )
    
    # 4. Plot Trendline
    plt.plot(diameters, p(diameters), "b--", alpha=0.5, 
             label=f'Exp. Fit ($R^2 = {r_squared:.4f}$)')

    # 5. Plot Simulated Data
    plt.plot(diameters, sim_forces, marker='s', linestyle='-', 
             label='Simulated Force', color='red', alpha=0.8)
    
    # --- AXIS CONSTRAINTS START HERE ---
    
    # Set X-axis to start at 0 and go slightly past max diameter
    plt.xlim(left=0, right=max(diameters) * 1.1)
    
    # Set Y-axis to start at 0 and go slightly past max force
    max_y = max(max(exp_forces + exp_cis), max(sim_forces))
    plt.ylim(bottom=0, top=max_y * 1.1)
    
    # --- AXIS CONSTRAINTS END HERE ---

    # Formatting
    plt.title(f'Force vs. Diameter Sweep (Pressure: {pressure} Bar, Dist: {distance}mm)')
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Resulting Force [N]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_size_sweep2(diameters, batch_experimental_averages, batch_experimental_cis, simulated_forces, pressure, distance):
    """
    Plots force vs. diameter, connecting experimental points with lines,
    and calculating R-squared based on simulation accuracy vs. experiment.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Convert to numpy arrays
    diameters = np.array(diameters)
    exp_forces = np.array(batch_experimental_averages)
    exp_cis = np.array(batch_experimental_cis)
    sim_forces = np.array(simulated_forces)

    # 2. Calculate R-squared (How well Simulation fits Experiment)
    # Residuals = Experimental Observed - Simulated Predicted
    residuals = exp_forces - sim_forces
    ss_res = np.sum(residuals**2)
    
    # Total sum of squares (variance in the experimental data)
    ss_tot = np.sum((exp_forces - np.mean(exp_forces))**2)
    
    # R^2 calculation
    r_squared = 1 - (ss_res / ss_tot)

    # 3. Plot Experimental Data (fmt='o-' connects the dots)
    plt.errorbar(
        diameters, exp_forces, yerr=exp_cis, 
        fmt='o-', capsize=5, label=f'Experimental (at {pressure} Bar)', 
        color='blue', markersize=8, linewidth=1.5
    )
    
    # 4. Plot Simulated Data
    # The R-squared is displayed here to show the "goodness of fit" of the sim
    plt.plot(
        diameters, sim_forces, marker='s', linestyle='--', 
        label=f'Simulated Force ($R^2 = {r_squared:.4f}$)', 
        color='red', alpha=0.8
    )
    
    # --- AXIS CONSTRAINTS ---
    plt.xlim(left=0, right=max(diameters) * 1.1)
    max_y = max(max(exp_forces + exp_cis), max(sim_forces))
    plt.ylim(bottom=0, top=max_y * 1.1)

    # Formatting
    plt.title(f'Force vs. Diameter: Simulation vs. Experiment\n(Pressure: {pressure} Bar, Dist: {distance}mm)')
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Resulting Force [N]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_3d_force_surface(pressures, diameters, batch_force_averages, type):
    """
    Plots a 3D surface of Force vs. Pressure and Diameter.
    
    pressures: 1D array/list of pressures
    diameters: 1D array/list of diameters
    batch_force_averages: 2D list/array where [diameter_index][pressure_index] = force
    """
    # 1. Prepare Data
    # Convert to numpy arrays for meshgrid compatibility
    P = np.array(pressures)
    D = np.array(diameters)
    Z = np.array(batch_force_averages)
    
    # Create the grid for X and Y axes
    # X will be Pressures, Y will be Diameters
    X, Y = np.meshgrid(P, D)

    # 2. Setup Figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot Surface
    # cmap='viridis' provides a clear color gradient for force magnitude
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # 4. Add Color Bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Resulting Force [N]')

    # 5. Labels and Title
    ax.set_xlabel('Nozzle Pressure [Bar]')
    ax.set_ylabel('Workpiece Diameter [mm]')
    ax.set_zlabel('Force [N]')
    ax.set_title('3D Force Analysis: Pressure vs. Diameter, '+type)

    # 6. Adjust View Angle (Optional)
    ax.view_init(elev=30, azim=225) # Adjust these numbers to rotate the view

    

    plt.tight_layout()
    #fig.subplots_adjust(left=0, right=0, bottom=0, top=0)

# --- Save Logic ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = f"3D_plot_{type}.png"
    full_file_path = os.path.join(save_path, filename)
    
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"3D plot saved to: {full_file_path}")

    plt.show()

def plot_3d_force_surface_residuals(pressures, diameters, batch_force_averages, sim_batch, type):
    """
    Plots a 3D surface of Force vs. Pressure and Diameter.
    
    pressures: 1D array/list of pressures
    diameters: 1D array/list of diameters
    batch_force_averages: 2D list/array where [diameter_index][pressure_index] = force
    """
    # 1. Prepare Data
    # Convert to numpy arrays for meshgrid compatibility
    P = np.array(pressures)
    D = np.array(diameters)
    exp = np.array(batch_force_averages)
    sim = np.array(sim_batch)
    Z = exp-sim
    
    # Create the grid for X and Y axes
    # X will be Pressures, Y will be Diameters
    X, Y = np.meshgrid(P, D)

    # 2. Setup Figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot Surface
    # cmap='viridis' provides a clear color gradient for force magnitude
    surf = ax.plot_surface(X, Y, Z, cmap='YlOrRd', edgecolor='none', alpha=0.8)
    
    # 4. Add Color Bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Resulting Force [N]')

    # 5. Labels and Title
    ax.set_xlabel('Nozzle Pressure [Bar]')
    ax.set_ylabel('Workpiece Diameter [mm]')
    ax.set_zlabel('Force [N]')
    ax.set_title('3D Force Residuals Analysis: Pressure vs. Diameter, '+type)

    # 6. Adjust View Angle (Optional)
    ax.view_init(elev=30, azim=225) # Adjust these numbers to rotate the view

    

    plt.tight_layout()
    #fig.subplots_adjust(left=0, right=0, bottom=0, top=0)

# --- Save Logic ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = f"3D_plot_{type}.png"
    full_file_path = os.path.join(save_path, filename)
    
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"3D plot saved to: {full_file_path}")

    plt.show()

def plot_3d_force_surface_residuals2(pressures, diameters, batch_force_averages, sim_batch, type):
    # 1. Prepare Data
    P = np.array(pressures)
    D = np.array(diameters)
    exp = np.array(batch_force_averages)
    sim = np.array(sim_batch)
    Z = exp - sim
    
    X, Y = np.meshgrid(P, D)

    # 2. Setup Figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- NEW COLOR LOGIC ---
    # Define the colors: Red (negative) -> Yellow -> Green (zero) -> Yellow -> Red (positive)
    # This ensures that "further from zero" in either direction becomes red.
    colors = ["purple", "blue", "green", "yellow", "red"]
    custom_cmap = LinearSegmentedColormap.from_list("residual_map", colors)
    
    # We use TwoSlopeNorm to force the center of the colormap (green) to be at 0.
    # We find the max absolute value to keep the scale somewhat balanced.
    vabs = max(abs(Z.min()), abs(Z.max()))
    # If all residuals are 0, we avoid a crash by setting a default range
    if vabs == 0: vabs = 1 
    
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    # -----------------------

    # 3. Plot Surface
    surf = ax.plot_surface(X, Y, Z, cmap=custom_cmap, norm=norm, edgecolor='none', alpha=0.8)

    # --- NEW: Project 0N Reference Plane ---
    # Create a Z array of zeros with the same shape as X and Y
    zero_plane = np.zeros_like(Z)
    
    # Plot the zero plane; alpha=0.2 makes it very transparent
    ax.plot_surface(X, Y, zero_plane, color='gray', alpha=0.2, shade=False)
    
    # Optional: Add a wireframe or a single line at 0 on the colorbar axis
    # to make the intersection more "crisp"
    ax.contour(X, Y, Z, levels=[0], colors='black', linestyles='dashed', linewidths=1)
    
    # 4. Add Color Bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Residual Error [N]')

    # 5. Labels and Title
    ax.set_xlabel('Nozzle Pressure [Bar]')
    ax.set_ylabel('Workpiece Diameter [mm]')
    ax.set_zlabel('Force Residual [N]')
    ax.set_title(f'3D Force Residuals Analysis: Pressure vs. Diameter, {type}')

    # 6. Adjust View Angle
    ax.view_init(elev=50, azim=225)

    plt.tight_layout()

    # --- Save Logic ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = f"3D_plot_{type}.png"
    full_file_path = os.path.join(save_path, filename)
    
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"3D plot saved to: {full_file_path}")

    plt.show()

def plot_mesh_comparison(pressures, batch_force_averages, batch_force_cis, 
                         batch_force_1mm, batch_force_cis_1mm, 
                         batch_force_5mm, batch_force_cis_5mm, 
                         diameter_index, diameters):
    """
    Plots the force comparison for different mesh sizes at a specific diameter index.
    """
    plt.figure(figsize=(10, 6))

    # Extract the data for the specific diameter
    y_default = batch_force_averages[diameter_index]
    ci_default = batch_force_cis[diameter_index]

    y_1mm = batch_force_1mm[diameter_index]
    ci_1mm = batch_force_cis_1mm[diameter_index]

    y_5mm = batch_force_5mm[diameter_index]
    ci_5mm = batch_force_cis_5mm[diameter_index]

    # Plot Default Batch
    plt.errorbar(pressures, y_default, yerr=ci_default, fmt='-o', capsize=4, 
                 label='No Mesh', color='blue', markersize=5, alpha=0.8)

    # Plot 1mm Batch
    plt.errorbar(pressures, y_1mm, yerr=ci_1mm, fmt='-s', capsize=4, 
                 label='1 mm Mesh Distance', color='green', markersize=5, alpha=0.8)

    # Plot 5mm Batch
    plt.errorbar(pressures, y_5mm, yerr=ci_5mm, fmt='-^', capsize=4, 
                 label='5 mm Mesh Distance', color='red', markersize=5, alpha=0.8)


    plt.xlim(left=0, right=max(pressures) * 1.1)
    plt.ylim(bottom=0, top=0.9)

    # Formatting
    plt.title(f'Force vs. Pressure, Workpiece Diameter: {diameters[diameter_index]} mm')
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Resulting Force [N]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Define a subfolder name (optional, e.g., 'output_plots')
    save_path = os.path.join(script_dir, "output_plots")
    
    # 3. Create the folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 4. Save the file
    filename = f"mesh_comparison_D{diameters[diameter_index]}.png"
    full_file_path = os.path.join(save_path, filename)
    
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to: {full_file_path}")

    plt.show()

def plot_size_sweep_mesh_compare(diameters, exp_forces, exp_cis, 
                                exp_forces_1mm, exp_cis_1mm, 
                                exp_forces_5mm, exp_cis_5mm, 
                                pressure_val, distance):
    """
    Plots Force vs. Diameter for three different mesh sizes at a fixed pressure.
    Connects points with lines (no fits) and saves to the script's folder.
    """
    plt.figure(figsize=(10, 6))

    # 1. Plot Default Mesh
    plt.errorbar(diameters, exp_forces, yerr=exp_cis, 
                 fmt='-o', capsize=5, label='No Mesh', 
                 color='blue', markersize=6, alpha=0.8)

    # 2. Plot 1mm Mesh
    plt.errorbar(diameters, exp_forces_1mm, yerr=exp_cis_1mm, 
                 fmt='-s', capsize=5, label='1 mm Mesh Distance', 
                 color='green', markersize=6, alpha=0.8)

    # 3. Plot 5mm Mesh
    plt.errorbar(diameters, exp_forces_5mm, yerr=exp_cis_5mm, 
                 fmt='-^', capsize=5, label='5 mm Mesh Distance', 
                 color='red', markersize=6, alpha=0.8)

    plt.xlim(left=0, right=max(diameters) * 1.1)
    plt.ylim(bottom=0)

    # Formatting
    plt.title(f'Force vs. Workpiece Diameter at {pressure_val} Bar')
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Resulting Force [N]')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Set axes to start at 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()

    # --- Save Logic ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = f"size_sweep_mesh_compare_{pressure_val}bar.png"
    full_file_path = os.path.join(save_path, filename)
    
    plt.savefig(full_file_path, dpi=300, bbox_inches='tight')
    print(f"Size sweep comparison saved to: {full_file_path}")

    plt.show()

def plot_2d_force_comparison_pressure(pressures, diameters, batch_force_averages, batch_force_cis, type=""):
    """
    Plots Force vs. Pressure as a 2D graph with error bars.
    Uses a colormap to handle many diameter curves uniquely.
    """
    plt.figure(figsize=(12, 8))
    
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    
    # 1. GENERATE UNIQUE COLORS
    # 'viridis' or 'plasma' are great because they are perceptually uniform
    # We sample the map at len(diameters) intervals
    colors = plt.cm.jet(np.linspace(0, 1, len(diameters)))
    
    for i, d in enumerate(diameters):
        forces_at_diameter = Z[i, :]
        errors_at_diameter = E[i, :]
        
        # 2. APPLY THE COLOR
        plt.errorbar(pressures, forces_at_diameter, yerr=errors_at_diameter, 
                     marker='o', capsize=3, label=f'D: {d}', 
                     alpha=0.8, color=colors[i]) # Assigned here

    plt.title('Force vs. Nozzle Pressure for Multiple Diameters, '+type)
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Resulting Force [N] (Avg ± 95% CI)')
    
    # The legend will now be much easier to read with the gradient
    plt.legend(title="Diameter [mm]", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_pressure_sweep_errors.png"), dpi=300)
    plt.show()

def plot_2d_force_comparison_dia(pressures, diameters, batch_force_averages, batch_force_cis, type=""):
    """
    Plots Force vs. Diameter as a 2D graph with error bars.
    Each line represents a different Nozzle Pressure.
    """
    plt.figure(figsize=(12, 8))
    
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    
    # 1. Setup the color gradient for pressures
    colors = plt.cm.plasma_r(np.linspace(0.1, 1.0, len(pressures)))
    
    # Iterate through each pressure to create a separate line
    for j, p in enumerate(pressures):
        forces_at_pressure = Z[:, j]
        errors_at_pressure = E[:, j]
        
        plt.errorbar(diameters, forces_at_pressure, yerr=errors_at_pressure, 
                     marker='o', capsize=3, label=f'{p}', 
                     alpha=0.8, color=colors[j])

    plt.title('Force vs. Workpiece Diameter for Multiple Pressures, ' + type)
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Resulting Force [N] (Avg ± 95% CI)')
    
    # 2. REVERSE THE LEGEND ORDER
    # Get current handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Pass them to the legend in reverse order [::-1]
    plt.legend(handles[::-1], labels[::-1], title="Pressure [bar]", 
               bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_diameter_sweep_errors.png"), dpi=300)
    plt.show()


def plot_2d_force_comparison_pressure_sim(pressures, diameters, batch_force_averages, type=""):
    """
    Plots Force vs. Pressure as a 2D graph with error bars.
    Uses a colormap to handle many diameter curves uniquely.
    """
    plt.figure(figsize=(12, 8))
    
    Z = np.array(batch_force_averages) 

    
    # 1. GENERATE UNIQUE COLORS
    # 'viridis' or 'plasma' are great because they are perceptually uniform
    # We sample the map at len(diameters) intervals
    colors = plt.cm.jet(np.linspace(0, 1, len(diameters)))
    
    for i, d in enumerate(diameters):
        forces_at_diameter = Z[i, :]
        
        
        # 2. APPLY THE COLOR
        plt.plot(pressures, forces_at_diameter, 
                 marker='o', 
                 markersize=4,
                 label=f'D: {d}', 
                 alpha=0.8, 
                 color=colors[i],
                 linewidth=1.5) # Assigned here

    plt.title('Simulated Force vs. Nozzle Pressure for Multiple Diameters, '+type)
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Simulated Force [N]')
    
    # The legend will now be much easier to read with the gradient
    plt.legend(title="Diameter [mm]", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_pressure_sweep_sim.png"), dpi=300)
    plt.show()

def plot_2d_force_comparison_dia_sim(pressures, diameters, batch_force_averages, type=""):
    """
    Plots Force vs. Diameter as a 2D graph with error bars.
    Each line represents a different Nozzle Pressure.
    """
    plt.figure(figsize=(12, 8))
    
    Z = np.array(batch_force_averages) 

    
    # 1. Setup the color gradient for pressures
    colors = plt.cm.plasma_r(np.linspace(0.1, 1.0, len(pressures)))
    
    # Iterate through each pressure to create a separate line
    for j, p in enumerate(pressures):
        forces_at_pressure = Z[:, j]

        
        plt.plot(diameters, forces_at_pressure, 
                 marker='o', 
                 markersize=4,
                 label=f'{p}', 
                 alpha=0.8, 
                 color=colors[j],
                 linewidth=1.5)

    plt.title('Simulated Force vs. Workpiece Diameter for Multiple Pressures, ' + type)
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Simulated Force [N]')
    
    # 2. REVERSE THE LEGEND ORDER
    # Get current handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Pass them to the legend in reverse order [::-1]
    plt.legend(handles[::-1], labels[::-1], title="Pressure [bar]", 
               bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_diameter_sweep_sim.png"), dpi=300)
    plt.show()



def plot_2d_force_comparison_pressure_sim2(pressures, diameters, batch_force_averages, batch_force_cis, sim_force, type=""):
    """
    Plots Force vs. Pressure as a 2D graph with error bars.
    Uses a colormap to handle many diameter curves uniquely.
    """
    plt.figure(figsize=(12, 8))
    
    Y = np.array(sim_force) 
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)

    
    # 1. GENERATE UNIQUE COLORS
    # 'viridis' or 'plasma' are great because they are perceptually uniform
    # We sample the map at len(diameters) intervals
    colors = plt.cm.jet(np.linspace(0, 1, len(diameters)))
    
    for i, d in enumerate(diameters):
        forces_at_diameter_sim = Y[i, :]
        forces_at_diameter = Z[i, :]
        errors_at_diameter = E[i, :]
        
        # 2. APPLY THE COLOR
        plt.plot(pressures, forces_at_diameter_sim, 
                 marker='o', 
                 linestyle='--',
                 markersize=4,
                 label=f'D: {d} (sim)', 
                 alpha=0.8, 
                 color=colors[i],
                 linewidth=1.5) # Assigned here
        
        plt.errorbar(pressures, forces_at_diameter, yerr=errors_at_diameter, 
                     marker='o', capsize=3, label=f'D: {d} (exp)', 
                     alpha=0.8, color=colors[i]) # Assigned here

    plt.title('Simulated and Experimental Force vs. Nozzle Pressure for Multiple Diameters, '+type)
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Force [N] (Avg ± 95% CI)')
    
    # The legend will now be much easier to read with the gradient
    plt.legend(title="Diameter [mm]", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_pressure_sweep_sim2.png"), dpi=300)
    plt.show()

def plot_2d_force_comparison_dia_sim2(pressures, diameters, batch_force_averages, batch_force_cis, sim_force, type=""):
    """
    Plots Force vs. Diameter as a 2D graph with error bars.
    Each line represents a different Nozzle Pressure.
    """
    plt.figure(figsize=(12, 8))
    
    Y = np.array(sim_force) 
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    
    # 1. Setup the color gradient for pressures
    colors = plt.cm.plasma_r(np.linspace(0.1, 1.0, len(pressures)))
    
    # Iterate through each pressure to create a separate line
    for j, p in enumerate(pressures):
        forces_at_pressure_sim = Y[:, j]
        forces_at_pressure = Z[:, j]
        errors_at_pressure = E[:, j]

        
        plt.plot(diameters, forces_at_pressure_sim, 
                 marker='o', 
                 linestyle='--',
                 markersize=4,
                 label=f'{p} (sim)', 
                 alpha=0.8, 
                 color=colors[j],
                 linewidth=1.5)
        
        plt.errorbar(diameters, forces_at_pressure, yerr=errors_at_pressure, 
                     marker='o', capsize=3, label=f'{p} (exp)', 
                     alpha=0.8, color=colors[j])

    plt.title('Simulated  and Experimental Force vs. Workpiece Diameter for Multiple Pressures, ' + type)
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Simulated Force [N] (Avg ± 95% CI)')
    
    # 2. REVERSE THE LEGEND ORDER
    # Get current handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Pass them to the legend in reverse order [::-1]
    plt.legend(handles[::-1], labels[::-1], title="Pressure [bar]", 
               bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_diameter_sweep_sim2.png"), dpi=300)
    plt.show()


def plot_pressure_nmbe(pressures, metrics1, metrics2, type_label=""):
    """
    Plots pre-calculated NMBE values against pressure.
    
    pressures: List or array of nozzle pressures (Bar)
    nmbe_values: List or array of pre-calculated NMBE percentages
    type_label: String to identify the data set (e.g., "Standard Model")
    """
    metric_index = 2
    metric1 = []
    metric2 = []
    for i in range(len(pressures)):
        metric1.append(metrics1[i][metric_index])
        metric2.append(metrics2[i][metric_index])


    plt.figure(figsize=(10, 6))
    
    # Plotting the NMBE line
    plt.plot(pressures, metric1, marker='o', linestyle='-', 
             color='tab:blue', linewidth=2, label=f'NMBE with new Fd {type_label}')
    
    plt.plot(pressures, metric2, marker='o', linestyle='-', 
             color='tab:red', linewidth=2, label=f'NMBE with old Fd {type_label}')
    
    # Add a horizontal line at 0 to clearly show over/under prediction
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Formatting
    plt.title(f'Normalized Mean Bias Error (NMBE) vs. Pressure {type_label}')
    plt.xlabel('Nozzle Pressure (Bar)')
    plt.ylabel('NMBE (%)')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Set y-axis to be symmetric around 0 if needed, or let it auto-scale
    limit = max(abs(np.array(metric2))) * 1.2
    plt.ylim(-limit, limit)
    
    plt.legend()
    plt.tight_layout()

    # Save Logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = f"NMBE_pressure_plot_{type_label.replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    print(f"NMBE plot saved to: {os.path.join(save_path, filename)}")
    
    plt.show()

def plot_workpiece_nmbe(diameters, metrics1, metrics2, type_label=""):
    """
    Plots pre-calculated NMBE values against pressure.
    
    pressures: List or array of nozzle pressures (Bar)
    nmbe_values: List or array of pre-calculated NMBE percentages
    type_label: String to identify the data set (e.g., "Standard Model")
    """
    metric_index = 2
    metric1 = []
    metric2 = []
    for i in range(len(diameters)):
        metric1.append(metrics1[i][metric_index])
        metric2.append(metrics2[i][metric_index])


    plt.figure(figsize=(10, 6))
    
    # Plotting the NMBE line
    plt.plot(diameters, metric1, marker='o', linestyle='-', 
             color='tab:blue', linewidth=2, label=f'NMBE with centricity coefficient {type_label}')
    
    plt.plot(diameters, metric2, marker='o', linestyle='-', 
             color='tab:red', linewidth=2, label=f'NMBE without centricity coefficient {type_label}')
    
    # Add a horizontal line at 0 to clearly show over/under prediction
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Formatting
    plt.title(f'Normalized Mean Bias Error (NMBE) vs. Diameter {type_label}')
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('NMBE [%]')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Set y-axis to be symmetric around 0 if needed, or let it auto-scale
    limit = max(abs(np.array(metric2))) * 1.2
    plt.ylim(-limit, limit)
    
    plt.legend()
    plt.tight_layout()

    # Save Logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    filename = f"NMBE_pressure_plot_{type_label.replace(' ', '_')}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    print(f"NMBE plot saved to: {os.path.join(save_path, filename)}")
    
    plt.show()

def plot_2d_force_comparison_pressure_with_sim(pressures, diameters, batch_force_averages, batch_force_cis, batch_force_simulations):
    """
    Plots Force vs. Pressure with error bars, overlaid with simulated data.
    Experimental data is solid; Simulated data is dashed.
    Each color corresponds to a different Workpiece Diameter.
    """
    plt.figure(figsize=(14, 8))
    
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    S = np.array(batch_force_simulations)
    
    # Iterate through each diameter index to create series
    for i, d in enumerate(diameters):
        # 1. Extract experimental and error data (row i)
        forces_at_diameter = Z[i, :]
        errors_at_diameter = E[i, :]
        
        # 2. Plot Experimental Data with Error Bars
        # We capture the line color generated here to reuse it for the simulation
        line = plt.errorbar(pressures, forces_at_diameter, yerr=errors_at_diameter, 
                             marker='o', capsize=3, linestyle='-', linewidth=2,
                             label=f'Exp. D: {d} mm', alpha=0.9)
        color = line[0].get_color()  # Get color of the experimental line

        # 3. Extract and Plot Simulated Data (row i)
        sim_at_diameter = S[i, :]
        
        # Plot with same color, but dashed linestyle and different marker
        plt.plot(pressures, sim_at_diameter, marker='s', markersize=4, linestyle='--', 
                 linewidth=1.5, color=color, label=f'Sim. D: {d} mm', alpha=0.7)

    # Plot Formatting
    plt.title('2D Force Comparison: Pressure Sweep vs. Simulation (Multiple Diameters)')
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Resulting Force [N] (Avg ± 95% CI)')
    
    # Place legend outside due to many lines
    plt.legend(title="Data Series", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_pressure_sweep_with_sim.png"), dpi=300)
    plt.show()

def plot_2d_force_comparison_dia_with_sim(pressures, diameters, batch_force_averages, batch_force_cis, batch_force_simulations):
    """
    Plots Force vs. Diameter with error bars, overlaid with simulated data.
    Experimental data is solid; Simulated data is dashed.
    Each color corresponds to a different Nozzle Pressure.
    """
    plt.figure(figsize=(14, 8))
    
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    S = np.array(batch_force_simulations)
    
    # Iterate through each pressure index to create series
    for j, p in enumerate(pressures):
        # 1. Extract experimental and error data (column j)
        forces_at_pressure = Z[:, j]
        errors_at_pressure = E[:, j]
        
        # 2. Plot Experimental Data with Error Bars
        line = plt.errorbar(diameters, forces_at_pressure, yerr=errors_at_pressure, 
                             marker='o', capsize=3, linestyle='-', linewidth=2,
                             label=f'Exp. {p} Bar', alpha=0.9)
        color = line[0].get_color() # Get color of the experimental line

        # 3. Extract and Plot Simulated Data (column j)
        sim_at_pressure = S[:, j]
        
        # Plot with same color, but dashed linestyle and different marker
        plt.plot(diameters, sim_at_pressure, marker='s', markersize=4, linestyle='--', 
                 linewidth=1.5, color=color, label=f'Sim. {p} Bar', alpha=0.7)

    # Plot Formatting
    plt.title('2D Force Comparison: Diameter Sweep vs. Simulation (Multiple Pressures)')
    plt.xlabel('Workpiece Diameter [mm]')
    plt.ylabel('Resulting Force [N] (Avg ± 95% CI)')
    
    # Place legend outside
    plt.legend(title="Data Series", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_force_diameter_sweep_with_sim.png"), dpi=300)
    plt.show()

def plot_2d_residuals_pressure(pressures, diameters, batch_force_averages, batch_force_cis, batch_force_simulations):
    """
    Plots the residuals (Experimental - Simulated) vs. Nozzle Pressure.
    A residual of 0 means perfect agreement.
    Positive = Simulation under-predicted; Negative = Simulation over-predicted.
    """
    plt.figure(figsize=(14, 8))
    
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    S = np.array(batch_force_simulations)
    
    # Calculate Residuals
    Residuals = Z - S
    
    # Iterate through each diameter to plot its specific residual curve
    for i, d in enumerate(diameters):
        res_at_diameter = Residuals[i, :]
        errors_at_diameter = E[i, :]
        
        plt.errorbar(
            pressures, res_at_diameter, yerr=errors_at_diameter, 
            marker='o', capsize=3, linestyle='-', linewidth=1.5,
            label=f'D: {d} mm', alpha=0.8
        )

    # Add a horizontal line at 0 to represent perfect simulation agreement
    plt.axhline(0, color='black', linewidth=2, linestyle='--', label='Perfect Agreement')
    
    # Formatting
    plt.title('Force Residuals (Exp - Sim) vs. Nozzle Pressure')
    plt.xlabel('Nozzle Pressure [Bar]')
    plt.ylabel('Force Residual [N] (Avg ± 95% CI)')
    
    # Dynamic Y-limits to ensure 0 is centered and all errors are visible
    limit = max(abs(Residuals.min() - E.max()), abs(Residuals.max() + E.max())) * 1.1
    plt.ylim(-limit, limit)
    
    plt.legend(title="Workpiece Diameter", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_residuals_pressure_sweep.png"), dpi=300)
    plt.show()

def plot_2d_residuals_dia(pressures, diameters, batch_force_averages, batch_force_cis, batch_force_simulations):
    """
    Plots the residuals (Experimental - Simulated) vs. Workpiece Diameter.
    Each line represents a different Nozzle Pressure.
    
    Positive Residual = Simulation under-predicted.
    Negative Residual = Simulation over-predicted.
    """
    plt.figure(figsize=(14, 8))
    
    # Convert to numpy for easy slicing [diameter_index, pressure_index]
    Z = np.array(batch_force_averages) 
    E = np.array(batch_force_cis)
    S = np.array(batch_force_simulations)
    
    # Calculate Residuals
    Residuals = Z - S
    
    # Iterate through each pressure index (columns) to create series
    for j, p in enumerate(pressures):
        # Extract column j: data for a specific pressure across all diameters
        res_at_pressure = Residuals[:, j]
        errors_at_pressure = E[:, j]
        
        plt.errorbar(
            diameters, res_at_pressure, yerr=errors_at_pressure, 
            marker='o', capsize=3, linestyle='-', linewidth=1.5,
            label=f'{p} Bar', alpha=0.8
        )

    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='black', linewidth=2, linestyle='--', label='Perfect Agreement')
    
    # Formatting
    plt.title('Force Residuals (Exp - Sim) vs. Workpiece Diameter')
    plt.xlabel('Workpiece Diameter [mm]')
    # Incorporating the 95% CI info here for clarity
    plt.ylabel('Force Residual [N] (Avg ± 95% CI)')
    
    # Center the Y-axis around zero
    limit = max(abs(Residuals.min() - E.max()), abs(Residuals.max() + E.max())) * 1.1
    plt.ylim(-limit, limit)
    
    plt.legend(title="Nozzle Pressure", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Save logic
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "2D_residuals_diameter_sweep.png"), dpi=300)
    plt.show()

def calculate_sim_force(pressure_range, flow_range, workpiece_name, distance, old_eq):
    sim_forces = []
    conv = 100000
    
    

    for i in range(0, len(pressure_range)):
        if(old_eq):
            
            results = calc_force(workpiece_name, [0,0,0], 1, distance, 4, pressure_range[i]*conv, 0, 2000, True, False, False, False)
            #print(flow_range[i])
        else:
            results = calc_force(workpiece_name, [0,0,0], 1, distance, 4, pressure_range[i]*conv, 0, 2000, False, False, False, False)
        
        sim_forces.append(results[0]*(results[3]**2))
    
    return sim_forces

def calculate_sim_force_batch(pressure_range, workpiece_names, distance, old_eq):
    sim_forces = []
    for i in range(0, len(workpiece_names)):
        sim_forces.append(calculate_sim_force(pressure_range, 0, workpiece_names[i], distance, old_eq))
    
    return sim_forces

def force_size_sweep(workpiece_names, pressure, distance):
    forces = []
    for i in range(0, len(workpiece_names)):
        results = calc_force(workpiece_names[i], [0,0,0], 1.17, distance, 4, pressure*100000, 0, 300, False, False, False, False)
        forces.append(results[0]*(results[3]**2))
    
    return forces

def flow_convert(flows, pressures, t1=293.15):
    
    output_flows = []
    roh=1.225
    p_amb = 101325
    

    for i in range(0, len(flows)):
        v_amb = flows[i]*0.1
        print(v_amb)
        m = v_amb*roh
        roh0 = (pressures[i]*100000+p_amb)/(287.05*t1)
        #print(roh0)
        v0 = m/roh0
        print(v0)
        output_flows.append(v0/0.1)

    return output_flows

def adjust_csv_values(input_file, output_file, add_value):
    """
    Reads a semicolon-delimited CSV, adds add_value to the second column 
    if the value is > 0, and saves to a new file.
    """
    try:
        with open(input_file, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=';')
            header = next(reader)  # Extract header
            
            modified_rows = [header]
            
            for row in reader:
                if len(row) < 2:
                    modified_rows.append(row)
                    continue
                
                # 1. Replace ',' with '.' to make it a valid float for Python
                # 2. Convert to float
                raw_val = row[1].replace(',', '.')
                try:
                    val = float(raw_val)
                    
                    # Apply logic: add value only if entry is > 0
                    if val > 0:
                        val += add_value
                    
                    # Convert back to string with ',' as decimal separator
                    # Using :E to maintain the Scientific Notation if desired
                    row[1] = "{:E}".format(val).replace('.', ',')
                except ValueError:
                    # Keep row as is if conversion fails (e.g. empty strings)
                    pass
                
                modified_rows.append(row)

        # Write to new CSV
        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=';')
            writer.writerows(modified_rows)
            
        print(f"Successfully processed. Saved to: {output_file}")

    except FileNotFoundError:
        print("Error: The input file was not found.")

def r_squared_batch(exp_batch, sim_batch):
    # 1. Convert to numpy arrays
    exp_forces = np.array(exp_batch)

    sim_forces = np.array(sim_batch)

    # 2. Calculate R-squared (How well Simulation fits Experiment)
    # Residuals = Experimental Observed - Simulated Predicted
    residuals = exp_forces - sim_forces
    ss_res = np.sum(residuals**2)
    
    # Total sum of squares (variance in the experimental data)
    ss_tot = np.sum((exp_forces - np.mean(exp_forces))**2)
    
    # R^2 calculation
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def calculate_validation_metrics(exp, sim):
    exp = np.array(exp)
    sim = np.array(sim)
    
    # Existing metrics
    mape = np.mean(np.abs((exp - sim) / exp)) * 100
    bias = (np.mean(sim) - np.mean(exp)) / np.mean(exp) * 100
    
    # R-squared calculation
    ss_res = np.sum((exp - sim)**2)
    ss_tot = np.sum((exp - np.mean(exp))**2)
    r2 = 1 - (ss_res / ss_tot)
    n = 96.0
    k = 2.0
    r2_adj = 1-(1-r2)*((n-1)/(n-k-1))
    
    #print(f"MAPE: {mape:.2f} %")
    #print(f"Bias: {bias:.2f} %")
    #print(f"R^2:  {r2:.4f}")
    
    return mape, bias, r2, r2_adj

def workpiece_metrics(exp, sim, diameters):
    results = []
    for i in range(len(exp)):
        mape, bias, r2, r2adj= calculate_validation_metrics(exp[i],sim[i])
        workpiece_result = diameters[i], round(mape,2), round(bias,2), round(r2,4)
        results.append(workpiece_result)
        print(workpiece_result)

    return results

def pressure_metrics(exp, sim, pressures):
    exp = np.array(exp)
    sim = np.array(sim)
    exp_t = exp.transpose()
    sim_t = sim.transpose()
    results = []
    for i in range(len(pressures)):
        mape, bias, r2, r2adj= calculate_validation_metrics(exp_t[i],sim_t[i])
        workpiece_result = pressures[i], round(mape,2), round(bias,2), round(r2,4)
        results.append(workpiece_result)
        print(workpiece_result)

    return results

def calculate_avg_relative_moe(data_points, ci_array):
    """
    Calculates the average relative margin of error (ratio of CI to data point).
    """
    data_points = np.asarray(data_points)
    ci_array = np.asarray(ci_array)
    
    # Calculate ratio for each point
    # Note: we use np.divide to handle potential zeros in data gracefully
    relative_errors = ci_array / data_points
    
    return np.mean(relative_errors)

def plot_force_vs_time(file_path, sampling_rate_hz=200):
    """
    Reads the CSV using the same logic as extract_averages_force, 
    but plots the raw force data over time.
    """
    time_data = []
    force_data = []
    
    # Read the file
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = list(csv.reader(f, delimiter=';'))
        
    row_count = len(reader)
    current_time = 0.0
    # Time step based on sampling rate (e.g., 1000Hz = 0.001s per row)
    dt = 1.0 / sampling_rate_hz 
    
    i = 0
    while i < 150000: #while i < row_count:
        try:
            # Match your extraction logic: European decimal (,) to float (.)
            value = float(reader[i][1].replace(',', '.'))
        except (IndexError, ValueError):
            i += 1
            current_time += dt
            continue

        
        force_data.append(value)
        time_data.append(current_time)
        i += 1
        current_time += dt
        

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.plot(time_data, force_data, color='blue', linewidth=1, label='Force Profile')
    
    # Formatting
    plt.title(f'Force vs. Time Profile\n40mm Workpiece Diameter')
    plt.xlabel('Time [s]')
    plt.ylabel('Force [N]')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    # Save logic...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "output_plots")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, "force_time_graph.png"), dpi=300)

    plt.show()
# config
#              0    1     2    3     4    5     6    7    8    9   10   11
pressures = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]
pressure_index = 0
distance = 25
testdata_filepath = '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-18_29_10.03.2026D40big.csv'

flow_data_filepath = '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/D40d26_2026-03-10.txt'
flow_data_raw, tank_pressures = extract_flow_data(flow_data_filepath)
flow_avgs, flow_cis = calculate_binned_stats(flow_data_raw)
nozzle_flows = flow_convert(flow_avgs, pressures)
sim_force_flow = calculate_sim_force(pressures, flow_avgs, '4dx1h_disc', distance, True)
#print(flow_avgs)
#print(nozzle_flows)
#print(sim_force_flow)
print(tank_pressures)
print(pressures)

#adjust_csv_values('/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/17_03_26-16_27_17.03.2026d35.csv',
#                  '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/17_03_26-16_27_17.03.2026d35rezero.csv',
#                  0.004)
#print(tank_pressures)
#for i in range(0, len(tank_pressures)):
    #print(pressures[i], tank_pressures[i])

#print(flow_avgs)



#print(nozzle_flows)

#print(flow_cis)
#print(sim_force_flow)

#             0    1    2     3     4     5     6     7
diameters = [#5.0, 
             7.5, 
             10.0, 
             15.0, 
             20.0, 
             25.0, 
             30.0, 
             35.0, 
             40.0] 
diameter_index = 7
workpiece_names = [ #'0_5dx1h_disc',
                   '0_75dx1h_disc',
                   '1dx1h_disc',
                   '1_5dx1h_disc',
                   '2dx1h_disc',
                   '2_5dx1h_disc',
                   '3dx1h_disc',
                   '3_5dx1h_disc',
                   '4dx1h_disc']
# removed: '0_5dx1h_disc'

testdata_filepaths = [
                        #'/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/10_03_26-16_49_10.03.2026D05d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-18_54_09.03.2026D075d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-18_24_09.03.2026D10d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-17_28_09.03.2026D15d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-17_12_09.03.2026D20d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-17_57_09.03.2026D25d26rezero.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/09_03_26-16_53_09.03.2026D30d26.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/17_03_26-16_27_17.03.2026d35rezero.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/10_03_26-16_07_10.03.2026D40d26.csv'
                      ]
#removed: '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/10_03_26-16_49_10.03.2026D05d26.csv'
#plot_force_vs_time('/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/10_03_26-16_07_10.03.2026D40d26.csv')
testdata_filepaths_mesh_1mm = [
                      #'/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_51_10.03.2026D05mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_36_10.03.2026D075mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_22_10.03.2026D10mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-20_08_10.03.2026D15mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_52_10.03.2026D20mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_37_10.03.2026D25mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_22_10.03.2026D30mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/17_03_26-16_43_17.03.2026d35mesh.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests/10_03_26-19_08_10.03.2026D40mesh.csv'
                      ]

testdata_filepaths_mesh_5mm = [
                      #'/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-17_28_16.03.2026d05mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-17_13_16.03.2026d075mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_58_16.03.2026 d10mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_41_16.03.2026d15_mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_26_16.03.2026d20mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-16_11_16.03.2026d25mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-15_53_16.03.2026d30mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/17_03_26-17_14_17.03.2026d35mesh_2.csv',
                      '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/mesh_tests2/16_03_26-15_33_16.03.2026d40mesh_2.csv'
                      ]


#testdata_filepath_smooth = '/Users/leonardolfens/Desktop/Python_Match/pybullet/testdata/misc/16_03_26-19_38_16.03.2026d30_smooth.csv'
#raw_force_averages_smooth = extract_averages_force(testdata_filepath_smooth)
#force_averages_smooth, force_stds_smooth = calculate_binned_stats(raw_force_averages_smooth)
#print(force_averages_smooth)
#print(raw_force_averages_smooth)

raw_force_averages = extract_averages_force(testdata_filepaths[diameter_index])
force_averages, force_stds = calculate_binned_stats(raw_force_averages)
#print(force_averages)


#print(sim_forces)

raw_averages_batch = extract_averages_force_batch(testdata_filepaths)
batch_force_averages, batch_force_cis = calculate_binned_stats_batch(raw_averages_batch)
raw_averages_batch_1mm = extract_averages_force_batch(testdata_filepaths_mesh_1mm)
batch_force_averages_1mm, batch_force_cis_1mm = calculate_binned_stats_batch(raw_averages_batch_1mm)
raw_averages_batch_5mm = extract_averages_force_batch(testdata_filepaths_mesh_5mm)
batch_force_averages_5mm, batch_force_cis_5mm = calculate_binned_stats_batch(raw_averages_batch_5mm)

sim_forces = calculate_sim_force(pressures, -1, workpiece_names[diameter_index], distance, False)
sim_forces2 = calculate_sim_force(pressures, 0, workpiece_names[diameter_index], distance, True)
#forces_size_sweep = force_size_sweep(workpiece_names, pressures[pressure_index], distance)



forces_size_sweep = []
for i in range(0, len(workpiece_names)):
    results = calc_force(workpiece_names[i], [0,0,0], 1.17, distance, 4, pressures[pressure_index]*100000, 0, 2000, False, False, False, False)
    forces_size_sweep.append(results[0]*(results[3]**2))
    print('name: '+workpiece_names[i]+', force: '+str(results[0])+', Cc: '+str(results[3]))

#print(forces_size_sweep)
#print(batch_force_averages)
#print(batch_force_cis)
for j in range(0, len(pressures)):
    exp_forces_size_sweep = []
    exp_cis_size_sweep = []
    exp_forces_size_sweep_1mm = []
    exp_cis_size_sweep_1mm = []
    exp_forces_size_sweep_5mm = []
    exp_cis_size_sweep_5mm = []
    for i in range(0, len(workpiece_names)):
        exp_forces_size_sweep.append(batch_force_averages[i][j])
        exp_cis_size_sweep.append(batch_force_cis[i][j])
        exp_forces_size_sweep_1mm.append(batch_force_averages_1mm[i][j])
        exp_cis_size_sweep_1mm.append(batch_force_cis_1mm[i][j])
        exp_forces_size_sweep_5mm.append(batch_force_averages_5mm[i][j])
        exp_cis_size_sweep_5mm.append(batch_force_cis_5mm[i][j])

    
    
    #plot_size_sweep_mesh_compare(diameters, exp_forces_size_sweep, exp_cis_size_sweep, 
    #                            exp_forces_size_sweep_1mm, exp_cis_size_sweep_1mm, 
    #                            exp_forces_size_sweep_5mm, exp_cis_size_sweep_5mm, 
    #                            pressures[j], distance)

exp_forces_size_sweep = []
exp_cis_size_sweep = []
exp_forces_size_sweep_1mm = []
exp_cis_size_sweep_1mm = []
exp_forces_size_sweep_5mm = []
exp_cis_size_sweep_5mm = []
for i in range(0, len(workpiece_names)):
    exp_forces_size_sweep.append(batch_force_averages[i][pressure_index])
    exp_cis_size_sweep.append(batch_force_cis[i][pressure_index])
    exp_forces_size_sweep_1mm.append(batch_force_averages_1mm[i][pressure_index])
    exp_cis_size_sweep_1mm.append(batch_force_cis_1mm[i][pressure_index])
    exp_forces_size_sweep_5mm.append(batch_force_averages_5mm[i][pressure_index])
    exp_cis_size_sweep_5mm.append(batch_force_cis_5mm[i][pressure_index])


#plot_2d_force_comparison_dia(pressures, diameters, batch_force_averages_5mm, batch_force_cis_5mm, "5 mm Mesh Distance")
#plot_2d_force_comparison_pressure(pressures, diameters, batch_force_averages_5mm, batch_force_cis_5mm, "5 mm Mesh Distance")
#plot_size_sweep2(diameters, exp_forces_size_sweep, exp_cis_size_sweep, forces_size_sweep, pressures[pressure_index], distance)
#plot_size_sweep(diameters, exp_forces_size_sweep, exp_cis_size_sweep, forces_size_sweep, pressures[pressure_index], distance)

#print(raw_averages)
#print(f"Total initial points: {len(raw_force_averages)}")
#print(f"Binned Averages: {force_averages}")
#print(f"Binned Std Devs: {force_stds}")
#print(f"Simulated Forces: {sim_forces}")

#sim_forces = calculate_sim_force(pressures, workpiece_names[diameter_index], distance)

#plot_results(pressures, force_averages, force_stds, sim_force_flow, diameters[diameter_index], distance)

#plot_results_sim_compare(pressures, force_averages, force_stds, sim_forces, sim_forces2, diameters[diameter_index], distance)

#plot_3d_force_surface(pressures, diameters, batch_force_averages, "No Mesh")
#plot_3d_force_surface(pressures, diameters, batch_force_averages_1mm, "1 mm Mesh Distance")
#plot_3d_force_surface(pressures, diameters, batch_force_averages_5mm, "5 mm Mesh Distance")

#if len(pressures) == len(force_averages) == len(sim_forces):
#    plot_results2(pressures, force_averages, force_stds, sim_forces, diameters[diameter_index], distance)
#    #plot_results(pressures, force_averages_smooth, force_stds_smooth, sim_forces, diameters[diameter_index], distance)
#else:
#    print(f"Error: Data length mismatch!")
#    print(f"Pressures: {len(pressures)}, Results: {len(force_averages)}, Sim: {len(sim_forces)}")

#calc_force('4dx1h_disc', [0,0,0], 1.17, 26, 4, 60000, 0, 300, True, True, True, False)
    
#plot_mesh_comparison(pressures, batch_force_averages, batch_force_cis, batch_force_averages_1mm, batch_force_cis_1mm, batch_force_averages_5mm, batch_force_cis_5mm, diameter_index, diameters)

#for i in range(0, len(diameters)):
#    plot_mesh_comparison(pressures, batch_force_averages, batch_force_cis, 
#                         batch_force_averages_1mm, batch_force_cis_1mm, 
#                         batch_force_averages_5mm, batch_force_cis_5mm, 
#                         i, diameters)
#for i in range(0, len(pressures)):      
#    plot_size_sweep_mesh_compare(diameters, exp_forces_size_sweep, exp_cis_size_sweep, 
#                                exp_forces_size_sweep_1mm, exp_cis_size_sweep_1mm, 
#                                exp_forces_size_sweep_5mm, exp_cis_size_sweep_5mm, 
#                                pressures[i], distance)
    
sim_force_batch = calculate_sim_force_batch(pressures, workpiece_names, 25, False)
#sim_force_batch_oldeq = calculate_sim_force_batch(pressures, workpiece_names, 25, True)

#plot_2d_force_comparison_pressure_with_sim(pressures, diameters, batch_force_averages, batch_force_cis, sim_force_batch)
#plot_2d_force_comparison_dia_with_sim(pressures, diameters, batch_force_averages, batch_force_cis, sim_force_batch)
#plot_2d_residuals_pressure(pressures, diameters, batch_force_averages_1mm, batch_force_cis_1mm, sim_force_batch)
#plot_2d_residuals_dia(pressures, diameters, batch_force_averages_1mm, batch_force_cis_1mm, sim_force_batch)

#plot_3d_force_surface(pressures, diameters, sim_force_batch, "Simulated without Mesh")
#plot_3d_force_surface_residuals2(pressures, diameters, batch_force_averages, sim_force_batch_oldeq, 'no mesh old eq')
#print(r_squared_batch(batch_force_averages_5mm, sim_force_batch))
#print(calculate_validation_metrics(batch_force_averages, sim_force_batch_oldeq))
sim_workpiece_metrics = workpiece_metrics(batch_force_averages_5mm, sim_force_batch, diameters)
#sim_workpiece_metrics_old_eq = workpiece_metrics(batch_force_averages, sim_force_batch_oldeq, diameters)

sim_pressure_metrics = pressure_metrics(batch_force_averages_5mm, sim_force_batch, pressures)
#sim_pressure_metrics = np.array(sim_pressure_metrics)
#sim_pressure_metrics_t = sim_pressure_metrics.transpose()
#print(sim_pressure_metrics_t[2])
#new_cds = []
#for i in range(len(sim_pressure_metrics_t[2])):
#    new_cds.append(100/(100+sim_pressure_metrics_t[2][i]))
#print(new_cds)

#sim_pressure_metrics_old_eq =  pressure_metrics(batch_force_averages, sim_force_batch_oldeq, pressures)
#plot_pressure_nmbe(pressures, sim_pressure_metrics, sim_pressure_metrics_old_eq)
#plot_workpiece_nmbe(diameters, sim_workpiece_metrics, sim_workpiece_metrics_old_eq)

#print(calculate_avg_relative_moe(batch_force_averages, batch_force_cis))
#print(calculate_avg_relative_moe(batch_force_averages_5mm, batch_force_cis_5mm))
#print(calculate_avg_relative_moe(batch_force_averages_1mm, batch_force_cis_1mm))

#plot_2d_force_comparison_pressure_sim(pressures, diameters, sim_force_batch, 'without Mesh Compensation')
#plot_2d_force_comparison_dia_sim(pressures, diameters, sim_force_batch, 'without Mesh Compensation')
    
#plot_2d_force_comparison_pressure_sim2(pressures, diameters, batch_force_averages_5mm, batch_force_cis_5mm, sim_force_batch, '5 mm Mesh Distance')
#plot_2d_force_comparison_dia_sim2(pressures, diameters, batch_force_averages_5mm, batch_force_cis_5mm, sim_force_batch, '5 mm Mesh Distance')