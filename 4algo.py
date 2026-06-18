import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# =====================================================
# 1. GLOBAL EXPERIMENT PARAMETERS
# =====================================================
GRID_SIZE = 100
NUM_NODES = 100       # Vehicles per map
NUM_STATIONS = 5      # 5 Directional Antennas
NUM_AGENTS = 50       # Swarm/Population size
MAX_ITER = 1000       # Exactly 1000 Iterations per run
TOTAL_RUNS = 100      # 100 random cases

# Geometric Binary Coverage Constraints
COVERAGE_RADIUS = 50
BEAMWIDTH = 90

dataset_filename = 'master_vehicle_maps.npy'
results_filename = 'final_comparison_results.csv'
graph_filename = 'optimization_results_graph.png'

# =====================================================
# 2. DATASET GENERATION & STORAGE
# =====================================================
if not os.path.exists(dataset_filename):
    print(f"Generating new master dataset of {TOTAL_RUNS} random cases...")
    master_maps = np.random.uniform(0, GRID_SIZE, size=(TOTAL_RUNS, NUM_NODES, 2))
    np.save(dataset_filename, master_maps)
    print(f"Saved strictly to: {dataset_filename}")
else:
    print(f"Loading existing master dataset from: {dataset_filename}")
    master_maps = np.load(dataset_filename)

# =====================================================
# 3. HIGH-SPEED VECTORIZED GEOMETRY ENGINE
# =====================================================
def evaluate_population(pop, nodes):
    """
    Evaluates all 50 agents simultaneously using Matrix Broadcasting.
    Returns an array of 50 fitness scores (Total Vehicles Covered).
    """
    pop_xy = pop[:, :, :2][:, :, np.newaxis, :]    
    nodes_xy = nodes[np.newaxis, np.newaxis, :, :] 
    
    diff = nodes_xy - pop_xy 
    dist = np.sqrt(diff[:, :, :, 0]**2 + diff[:, :, :, 1]**2) 
    
    angles = np.arctan2(diff[:, :, :, 1], diff[:, :, :, 0])
    station_angles = np.radians(pop[:, :, 2])[:, :, np.newaxis]
    
    angle_diff = (angles - station_angles + np.pi) % (2 * np.pi) - np.pi
    half_beam = np.radians(BEAMWIDTH) / 2
    
    in_sector = np.abs(angle_diff) <= half_beam
    in_radius = dist <= COVERAGE_RADIUS
    
    covered_by_station = in_sector & in_radius      
    covered_by_any = np.any(covered_by_station, axis=1) 
    
    return np.sum(covered_by_any, axis=1) 

# =====================================================
# 4. PARTICLE SWARM OPTIMIZATION (PSO)
# =====================================================
def run_pso(nodes):
    w, c1, c2 = 0.7, 1.5, 1.5
    pop = np.random.uniform(0, GRID_SIZE, size=(NUM_AGENTS, NUM_STATIONS, 3))
    pop[:, :, 2] = np.random.uniform(0, 360, size=(NUM_AGENTS, NUM_STATIONS))
    vel = np.zeros_like(pop)
    
    pbest_pos = pop.copy()
    pbest_scores = evaluate_population(pop, nodes)
    
    gbest_idx = np.argmax(pbest_scores)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    for _ in range(MAX_ITER):
        r1 = np.random.rand(NUM_AGENTS, NUM_STATIONS, 3)
        r2 = np.random.rand(NUM_AGENTS, NUM_STATIONS, 3)
        
        vel = w * vel + c1 * r1 * (pbest_pos - pop) + c2 * r2 * (gbest_pos - pop)
        pop += vel
        
        pop[:, :, 0] = np.clip(pop[:, :, 0], 0, GRID_SIZE)
        pop[:, :, 1] = np.clip(pop[:, :, 1], 0, GRID_SIZE)
        pop[:, :, 2] = pop[:, :, 2] % 360
        
        scores = evaluate_population(pop, nodes)
        
        improved = scores > pbest_scores
        pbest_scores[improved] = scores[improved]
        pbest_pos[improved] = pop[improved].copy()
        
        best_idx = np.argmax(pbest_scores)
        if pbest_scores[best_idx] > gbest_score:
            gbest_score = pbest_scores[best_idx]
            gbest_pos = pbest_pos[best_idx].copy()
            
    return gbest_score

# =====================================================
# 5. MOTH FLAME OPTIMIZATION (MFO)
# =====================================================
def run_mfo(nodes):
    b = 1 
    moths = np.random.uniform(0, GRID_SIZE, size=(NUM_AGENTS, NUM_STATIONS, 3))
    moths[:, :, 2] = np.random.uniform(0, 360, size=(NUM_AGENTS, NUM_STATIONS))
    
    flames = np.copy(moths)
    flame_scores = np.zeros(NUM_AGENTS)
    
    for it in range(MAX_ITER):
        moth_scores = evaluate_population(moths, nodes)
        
        if it == 0:
            sort_idx = np.argsort(moth_scores)[::-1] 
            flames = moths[sort_idx].copy()
            flame_scores = moth_scores[sort_idx].copy()
        else:
            combined_pop = np.vstack((moths, flames))
            combined_scores = np.concatenate((moth_scores, flame_scores))
            sort_idx = np.argsort(combined_scores)[::-1]
            flames = combined_pop[sort_idx[:NUM_AGENTS]].copy()
            flame_scores = combined_scores[sort_idx[:NUM_AGENTS]].copy()
            
        flame_no = round(NUM_AGENTS - it * ((NUM_AGENTS - 1) / MAX_ITER))
        a = -1 + it * (-1 / MAX_ITER) 
        
        for i in range(NUM_AGENTS):
            for j in range(NUM_STATIONS):
                for k in range(3):
                    flame_idx = i if i < flame_no else flame_no - 1
                    distance_to_flame = np.abs(flames[flame_idx, j, k] - moths[i, j, k])
                    t = (a - 1) * np.random.rand() + 1
                    moths[i, j, k] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + flames[flame_idx, j, k]
                    
        moths[:, :, 0] = np.clip(moths[:, :, 0], 0, GRID_SIZE)
        moths[:, :, 1] = np.clip(moths[:, :, 1], 0, GRID_SIZE)
        moths[:, :, 2] = moths[:, :, 2] % 360
        
    return flame_scores[0] 

# =====================================================
# 6. FIREFLY ALGORITHM (FA)
# =====================================================
def run_fa(nodes):
    pop = np.random.uniform(0, GRID_SIZE, size=(NUM_AGENTS, NUM_STATIONS, 3))
    pop[:, :, 2] = np.random.uniform(0, 360, size=(NUM_AGENTS, NUM_STATIONS))
    scores = evaluate_population(pop, nodes)
    
    best_score = np.max(scores)
    beta0 = 1.0     # Base attractiveness
    gamma = 0.01    # Light absorption coefficient
    
    for it in range(MAX_ITER):
        # Alpha decays over time to improve final convergence
        alpha_step = 2.0 * (1.0 - it / MAX_ITER) 
        new_pop = pop.copy()
        
        for i in range(NUM_AGENTS):
            moved = False
            for j in range(NUM_AGENTS):
                # If firefly J is brighter (covers more vehicles) than firefly I
                if scores[j] > scores[i]:
                    r = np.sum((pop[i] - pop[j])**2) 
                    beta = beta0 * np.exp(-gamma * r)
                    rand_step = alpha_step * (np.random.rand(NUM_STATIONS, 3) - 0.5)
                    
                    new_pop[i] += beta * (pop[j] - pop[i]) + rand_step
                    moved = True
            
            # If no one is brighter, wander randomly
            if not moved:
                new_pop[i] += alpha_step * (np.random.rand(NUM_STATIONS, 3) - 0.5)
                
        pop = new_pop
        pop[:, :, 0] = np.clip(pop[:, :, 0], 0, GRID_SIZE)
        pop[:, :, 1] = np.clip(pop[:, :, 1], 0, GRID_SIZE)
        pop[:, :, 2] = pop[:, :, 2] % 360
        
        scores = evaluate_population(pop, nodes)
        if np.max(scores) > best_score:
            best_score = np.max(scores)
            
    return best_score

# =====================================================
# 7. GREY WOLF OPTIMIZER (GWO)
# =====================================================
def run_gwo(nodes):
    pop = np.random.uniform(0, GRID_SIZE, size=(NUM_AGENTS, NUM_STATIONS, 3))
    pop[:, :, 2] = np.random.uniform(0, 360, size=(NUM_AGENTS, NUM_STATIONS))
    
    alpha_pos = np.zeros((NUM_STATIONS, 3))
    beta_pos = np.zeros((NUM_STATIONS, 3))
    delta_pos = np.zeros((NUM_STATIONS, 3))
    
    alpha_score, beta_score, delta_score = -1, -1, -1
    
    for it in range(MAX_ITER):
        scores = evaluate_population(pop, nodes)
        
        # Determine Alpha, Beta, Delta wolves (Top 3 solutions)
        for i in range(NUM_AGENTS):
            if scores[i] > alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = scores[i], pop[i].copy()
            elif scores[i] > beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = scores[i], pop[i].copy()
            elif scores[i] > delta_score:
                delta_score, delta_pos = scores[i], pop[i].copy()
                
        # "a" dictates the exploration/exploitation ratio (linearly drops 2 to 0)
        a = 2.0 - it * (2.0 / MAX_ITER)
        
        for i in range(NUM_AGENTS):
            # Alpha influence
            r1, r2 = np.random.rand(NUM_STATIONS, 3), np.random.rand(NUM_STATIONS, 3)
            A1, C1 = 2 * a * r1 - a, 2 * r2
            D_alpha = np.abs(C1 * alpha_pos - pop[i])
            X1 = alpha_pos - A1 * D_alpha
            
            # Beta influence
            r1, r2 = np.random.rand(NUM_STATIONS, 3), np.random.rand(NUM_STATIONS, 3)
            A2, C2 = 2 * a * r1 - a, 2 * r2
            D_beta = np.abs(C2 * beta_pos - pop[i])
            X2 = beta_pos - A2 * D_beta
            
            # Delta influence
            r1, r2 = np.random.rand(NUM_STATIONS, 3), np.random.rand(NUM_STATIONS, 3)
            A3, C3 = 2 * a * r1 - a, 2 * r2
            D_delta = np.abs(C3 * delta_pos - pop[i])
            X3 = delta_pos - A3 * D_delta
            
            # Omega wolves move based on the average of Alpha, Beta, Delta
            pop[i] = (X1 + X2 + X3) / 3.0
            
        pop[:, :, 0] = np.clip(pop[:, :, 0], 0, GRID_SIZE)
        pop[:, :, 1] = np.clip(pop[:, :, 1], 0, GRID_SIZE)
        pop[:, :, 2] = pop[:, :, 2] % 360
        
    # Final evaluation check for true alpha
    scores = evaluate_population(pop, nodes)
    return max(alpha_score, np.max(scores))

# =====================================================
# 8. BATCH EXECUTION & FILE SAVING
# =====================================================
results = []

print(f"\nBeginning Strict Benchmark: 4 Algorithms | {TOTAL_RUNS} Cases | {MAX_ITER} Iterations")
print("This will process silently in the background to maximize speed...")
print("-" * 75)

for run in range(TOTAL_RUNS):
    current_nodes = master_maps[run]
    
    # Process all 4 algorithms consecutively on the exact same dataset
    pso_score = run_pso(current_nodes)
    mfo_score = run_mfo(current_nodes)
    fa_score  = run_fa(current_nodes)
    gwo_score = run_gwo(current_nodes)
    
    results.append([run + 1, pso_score, mfo_score, fa_score, gwo_score])
    print(f"Case {run+1:>3}/{TOTAL_RUNS} | PSO: {pso_score:g} | MFO: {mfo_score:g} | FA: {fa_score:g} | GWO: {gwo_score:g}")

# Save numerical results strictly to CSV file in folder
with open(results_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Case_ID", "PSO_Coverage", "MFO_Coverage", "FA_Coverage", "GWO_Coverage"])
    writer.writerows(results)

print("\nData Collection Complete.")
print(f"Results CSV saved to: {results_filename}")

# =====================================================
# 9. GRAPH FORMATTING & GENERATION
# =====================================================
# Creating the graph headless (saved to file, no popup window/animations)
print("Generating final comparison graph...")
results_np = np.array(results)[:, 1:] 

plt.figure(figsize=(10, 7))

# Formatting the Boxplot
box = plt.boxplot([results_np[:, 0], results_np[:, 1], results_np[:, 2], results_np[:, 3]], 
                  labels=['Particle Swarm (PSO)', 'Moth Flame (MFO)', 'Firefly Algorithm (FA)', 'Grey Wolf (GWO)'], 
                  patch_artist=True)

# Graph styling and coloring
colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

for median in box['medians']:
    median.set(color='red', linewidth=2.5)

# Axis formatting
plt.title("Meta-Heuristic Optimization Comparison\n(Maximized Binary Coverage over 100 Random Cases)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Maximum Vehicles Covered (Max 100)", fontsize=12, fontweight='bold')
plt.xlabel("Optimization Algorithms", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 105)

# Save the plot securely to the folder
plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
print(f"Well-formatted graph successfully saved to: {graph_filename}")
print("-" * 75)