from vpython import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import math
import heapq

# VPython Simulation Setup
scene = canvas(title="UAV Drone with Manipulator Arm Simulation", width=800, height=600, background=color.white)
scene.range = 8
scene.center = vector(0, 1, 0)

# Ground
ground = box(
    pos=vector(0, -0.1, 0),
    size=vector(24, 0.2, 24),
    color=vector(0.13, 0.55, 0.13)
)

# Static obstacles
static_obstacles = [
    box(pos=vector(2, 1, 0), size=vector(1, 2, 1), color=color.gray(0.5)),
    box(pos=vector(-2, 1, 0), size=vector(1, 2, 1), color=color.gray(0.5)),
    box(pos=vector(1, 1, 5), size=vector(1, 2, 1), color=color.gray(0.5)),
    box(pos=vector(1, 1, -5), size=vector(1, 2, 1), color=color.gray(0.5)),
    box(pos=vector(7, 1, -3), size=vector(1, 2, 1), color=color.gray(0.5)),
    box(pos=vector(-7, 1, 3), size=vector(1, 2, 1), color=color.gray(0.5))
]

# Moving obstacles
moving_obstacle_lr1 = sphere(pos=vector(3, 1, 2), radius=0.5, color=color.orange)
moving_obstacle_lr2 = sphere(pos=vector(-3, 1, -2), radius=0.5, color=color.orange)
moving_obstacle_ud1 = sphere(pos=vector(4, 1.5, 0), radius=0.5, color=color.purple)
moving_obstacle_ud2 = sphere(pos=vector(-4, 1.5, 4), radius=0.5, color=color.purple)
moving_obstacles = [moving_obstacle_lr1, moving_obstacle_lr2, moving_obstacle_ud1, moving_obstacle_ud2]

# Fan Propeller 
def create_fan_propeller(center, num_blades=3):
    blade_length = 0.4
    blade_width = 0.05
    hub_radius = 0.05
    blades = []
    
    for i in range(num_blades):
        angle = i * (360 / num_blades)
        angle_rad = math.radians(angle)
        offset = vector((blade_length/2) * math.cos(angle_rad), 0, (blade_length/2) * math.sin(angle_rad))
        blade = box(pos=center + offset,
                    size=vector(blade_length, blade_width, blade_width),
                    color=color.white)
        blade.axis = vector(blade_length * math.cos(angle_rad), 0, blade_length * math.sin(angle_rad))
        blades.append(blade)
    
    hub = sphere(pos=center, radius=hub_radius, color=color.gray(0.5))
    fan = compound(blades + [hub], pos=center)
    return fan

# Drone body
drone_body = box(pos=vector(0, 1, 0),
                 size=vector(0.6, 0.2, 0.6),
                 color=vector(0.82, 0.71, 0.55))

propeller_offsets = [vector(0.3, 0.1, 0.3), vector(-0.3, 0.1, 0.3),
                     vector(0.3, 0.1, -0.3), vector(-0.3, 0.1, -0.3)]
propellers = [create_fan_propeller(drone_body.pos + offset) for offset in propeller_offsets]

# Arm and arm joint 
arm = cylinder(pos=drone_body.pos, axis=vector(0, -0.5, 0), radius=0.05, color=color.black)
arm_joint = sphere(pos=arm.pos + arm.axis, radius=0.1, color=color.black)

# Gripper
def create_finger(angle_offset):
    base_pos = arm_joint.pos
    angle_rad = math.radians(angle_offset)
    x_offset = 0.1 * math.cos(angle_rad)
    z_offset = 0.1 * math.sin(angle_rad)
    finger_base = cylinder(pos=base_pos + vector(x_offset, 0, z_offset),
                           axis=vector(0, -0.1, 0), radius=0.02, color=color.yellow)
    finger_curve = cylinder(pos=finger_base.pos + finger_base.axis,
                            axis=vector(x_offset * 0.5, -0.05, z_offset * 0.5), radius=0.02, color=color.yellow)
    finger_tip = cylinder(pos=finger_curve.pos + finger_curve.axis,
                          axis=vector(-x_offset * 0.5, -0.05, -z_offset * 0.5), radius=0.02, color=color.yellow)
    return [finger_base, finger_curve, finger_tip]

gripper = [{'angle': angle, 'parts': create_finger(angle)} for angle in [0, 120, 240]]

def update_gripper():
    for finger in gripper:
        angle_offset = finger['angle']
        base_pos = arm_joint.pos
        angle_rad = math.radians(angle_offset)
        x_offset = 0.1 * math.cos(angle_rad)
        z_offset = 0.1 * math.sin(angle_rad)
        finger['parts'][0].pos = base_pos + vector(x_offset, 0, z_offset)
        finger['parts'][0].axis = vector(0, -0.1, 0)
        finger['parts'][1].pos = finger['parts'][0].pos + finger['parts'][0].axis
        finger['parts'][1].axis = vector(x_offset * 0.5, -0.05, z_offset * 0.5)
        finger['parts'][2].pos = finger['parts'][1].pos + finger['parts'][1].axis
        finger['parts'][2].axis = vector(-x_offset * 0.5, -0.05, -z_offset * 0.5)

update_gripper()

# Object 
obj = sphere(pos=vector(5, 0, 0), radius=0.15, color=color.blue)

# Recording for UAV path, positions, and moving obstacles
uav_path = []  
uav_positions = []  
moving_obstacle_paths = [[], [], [], []]
grasped = False
t = 0

def update_moving_obstacles(dt):
    global t
    t += dt
    for i, obs in enumerate(moving_obstacles[:2]):
        base_x = 3 if i == 0 else -3
        obs.pos.x = base_x + 2 * math.sin(t * math.pi / 2)
        moving_obstacle_paths[i].append(vector(obs.pos.x, obs.pos.y, obs.pos.z))
    for i, obs in enumerate(moving_obstacles[2:], 2):
        base_y = 1.5
        obs.pos.y = base_y + 1 * math.cos(t * math.pi / 1.5)
        moving_obstacle_paths[i].append(vector(obs.pos.x, obs.pos.y, obs.pos.z))

# Movement Functions
def move_to(target, duration):
    global drone_body, propellers, t
    steps = int(duration * 100)
    dt = duration / steps
    start_pos = drone_body.pos
    for i in range(steps):
        rate(100)
        fraction = (i + 1) / steps
        new_pos = start_pos + fraction * (target - start_pos)
        drone_body.pos = new_pos
        uav_path.append(vector(new_pos.x, new_pos.y, new_pos.z))  # For 3D animation
        uav_positions.append(vector(new_pos.x, new_pos.y, new_pos.z))  # For 2D graph
        
        for j, offset in enumerate(propeller_offsets):
            propellers[j].pos = drone_body.pos + offset
            propellers[j].rotate(angle=0.2, axis=vector(0, 1, 0), origin=propellers[j].pos)
            
        arm.pos = drone_body.pos
        arm_joint.pos = arm.pos + arm.axis
        update_gripper()
        if grasped:
            obj.pos = arm_joint.pos
        
        update_moving_obstacles(dt)

def extend_arm(target_length, duration):
    global arm, grasped, arm_joint
    steps = int(duration * 100)
    dt = duration / steps
    start_length = arm.axis.y
    for i in range(steps):
        rate(100)
        fraction = (i + 1) / steps
        new_length = start_length + fraction * (target_length - start_length)
        arm.axis = vector(0, new_length, 0)
        arm_joint.pos = arm.pos + arm.axis
        update_gripper()
        if grasped:
            obj.pos = arm_joint.pos
        uav_path.append(vector(drone_body.pos.x, drone_body.pos.y, drone_body.pos.z))  # For 3D animation
        uav_positions.append(vector(drone_body.pos.x, drone_body.pos.y, drone_body.pos.z))  # For 2D graph
        update_moving_obstacles(dt)

def grasp():
    global grasped, obj
    grasped = True
    obj.pos = arm_joint.pos

def release(target_pos):
    global grasped, obj
    grasped = False
    obj.pos = target_pos

# Path Planning (A* Algorithm) for Static Obstacles
def plan_path_Astar(start, goal):
    resolution = 0.5
    margin = 1.5
    x_min, x_max = -10, 10
    z_min, z_max = -10, 10

    def coord_to_grid(pos):
        i = int((pos.x - x_min) / resolution)
        j = int((pos.z - z_min) / resolution)
        return (i, j)

    def grid_to_coord(cell):
        i, j = cell
        x = x_min + i * resolution + resolution / 2
        z = z_min + j * resolution + resolution / 2
        return vector(x, start.y, z)

    grid_width = int((x_max - x_min) / resolution) + 1
    grid_height = int((z_max - z_min) / resolution) + 1
    occupancy = [[False for j in range(grid_height)] for i in range(grid_width)]
    
    for i in range(grid_width):
        for j in range(grid_height):
            cell_center = grid_to_coord((i, j))
            for obs in static_obstacles:
                obs_half_x = obs.size.x / 2
                obs_half_z = obs.size.z / 2
                if (abs(cell_center.x - obs.pos.x) <= obs_half_x + margin and
                    abs(cell_center.z - obs.pos.z) <= obs_half_z + margin):
                    occupancy[i][j] = True
                    break

    start_cell = coord_to_grid(start)
    goal_cell = coord_to_grid(goal)

    open_heap = []
    heapq.heappush(open_heap, (0, start_cell))
    came_from = {}
    g_score = { (i, j): float('inf') for i in range(grid_width) for j in range(grid_height) }
    g_score[start_cell] = 0

    def heuristic(cell):
        cell_pos = grid_to_coord(cell)
        goal_pos = grid_to_coord(goal_cell)
        return mag(cell_pos - goal_pos)

    f_score = { (i, j): float('inf') for i in range(grid_width) for j in range(grid_height) }
    f_score[start_cell] = heuristic(start_cell)

    closed_set = set()
    while open_heap:
        current = heapq.heappop(open_heap)[1]
        if current == goal_cell:
            path = [grid_to_coord(current)]
            while current in came_from:
                current = came_from[current]
                path.append(grid_to_coord(current))
            path.reverse()
            return path

        closed_set.add(current)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                neighbor = (current[0] + di, current[1] + dj)
                if not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height):
                    continue
                if occupancy[neighbor[0]][neighbor[1]]:
                    continue
                if neighbor in closed_set:
                    continue
                tentative_g = g_score[current] + mag(grid_to_coord(current) - grid_to_coord(neighbor))
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    return [goal]

# Navigation Sequence
start_pos = vector(0, 1, 0)
pickup_point = vector(5, 1, 0)
dropoff_point = vector(-5, 0.2, 0)
above_pickup = pickup_point + vector(0, 2, 0)
above_dropoff = dropoff_point + vector(0, 2, 0)

astar_path1 = plan_path_Astar(start_pos, above_pickup)
for wp in astar_path1:
    move_to(wp, 3)
move_to(pickup_point, 3)
extend_arm(-0.8, 1)
grasp()
extend_arm(-0.5, 1)
move_to(above_pickup, 3)

astar_path2 = plan_path_Astar(above_pickup, above_dropoff)
for wp in astar_path2:
    move_to(wp, 3)
move_to(dropoff_point, 3)
extend_arm(-0.8, 1)
release(dropoff_point)
extend_arm(-0.5, 1)
move_to(above_dropoff, 3)

print("Simulation complete.")

# 2D Numerical Graph with UAV and Obstacle Positions
x = np.array([p.x for p in uav_positions])
y = np.array([p.y for p in uav_positions])
z = np.array([p.z for p in uav_positions])
lr1_x = np.array([p.x for p in moving_obstacle_paths[0]])
lr2_x = np.array([p.x for p in moving_obstacle_paths[1]])
ud1_y = np.array([p.y for p in moving_obstacle_paths[2]])
ud2_y = np.array([p.y for p in moving_obstacle_paths[3]])

total_steps = len(uav_positions)
dt = 0.01
time = np.linspace(0, total_steps * dt, total_steps)

def trim_or_pad(array, target_length):
    if len(array) > target_length:
        return array[:target_length]
    elif len(array) < target_length:
        return np.pad(array, (0, target_length - len(array)), mode='edge')
    return array

lr1_x = trim_or_pad(lr1_x, total_steps)
lr2_x = trim_or_pad(lr2_x, total_steps)
ud1_y = trim_or_pad(ud1_y, total_steps)
ud2_y = trim_or_pad(ud2_y, total_steps)

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(time, x, 'b-', label='UAV X')
ax1.plot(time, lr1_x, 'orange', linestyle='--', label='LR1 X')
ax1.plot(time, lr2_x, 'orange', linestyle='-.', label='LR2 X')
ax1.set_ylabel('X Position (units)')
ax1.legend()
ax1.grid(True)

ax2.plot(time, y, 'g-', label='UAV Y')
ax2.plot(time, ud1_y, 'purple', linestyle='--', label='UD1 Y')
ax2.plot(time, ud2_y, 'purple', linestyle='-.', label='UD2 Y')
ax2.set_ylabel('Y Position (units)')
ax2.legend()
ax2.grid(True)

ax3.plot(time, z, 'r-', label='UAV Z')
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Z Position (units)')
ax3.legend()
ax3.grid(True)

plt.suptitle("UAV and Obstacle Positions Over Time")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3D UAV Path Animation
x = np.array([p.x for p in uav_path])
y = np.array([p.y for p in uav_path])
z = np.array([p.z for p in uav_path])
lr1_x = np.array([p.x for p in moving_obstacle_paths[0]])
lr1_y = np.array([p.y for p in moving_obstacle_paths[0]])
lr1_z = np.array([p.z for p in moving_obstacle_paths[0]])
lr2_x = np.array([p.x for p in moving_obstacle_paths[1]])
lr2_y = np.array([p.y for p in moving_obstacle_paths[1]])
lr2_z = np.array([p.z for p in moving_obstacle_paths[1]])
ud1_x = np.array([p.x for p in moving_obstacle_paths[2]])
ud1_y = np.array([p.y for p in moving_obstacle_paths[2]])
ud1_z = np.array([p.z for p in moving_obstacle_paths[2]])
ud2_x = np.array([p.x for p in moving_obstacle_paths[3]])
ud2_y = np.array([p.y for p in moving_obstacle_paths[3]])
ud2_z = np.array([p.z for p in moving_obstacle_paths[3]])

static_obstacle_data = [
    {'pos': (obs.pos.x, obs.pos.y, obs.pos.z),
     'size': (obs.size.x, obs.size.y, obs.size.z)}
    for obs in static_obstacles
]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("UAV Path Animation with Moving Obstacles")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-8, 8)
ax.set_ylim(-1, 4)
ax.set_zlim(-8, 8)

ground_x = np.linspace(-12, 12, 10)
ground_z = np.linspace(-12, 12, 10)
ground_x, ground_z = np.meshgrid(ground_x, ground_z)
ground_y = np.full_like(ground_x, -0.1)
ax.plot_surface(ground_x, ground_y, ground_z, color='green', alpha=0.3)

for obs in static_obstacle_data:
    ox, oy, oz = obs['pos']
    sx, sy, sz = obs['size']
    r = [-sx/2, sx/2]
    s = [-sy/2, sy/2]
    t_list = [-sz/2, sz/2]
    for i in range(2):
        for j in range(2):
            ax.plot3D([ox+r[i], ox+r[i]], [oy+s[0], oy+s[1]], [oz+t_list[j], oz+t_list[j]], color='gray')
            ax.plot3D([ox+r[i], ox+r[i]], [oy+s[j], oy+s[j]], [oz+t_list[0], oz+t_list[1]], color='gray')
            ax.plot3D([ox+r[0], ox+r[1]], [oy+s[i], oy+s[i]], [oz+t_list[j], oz+t_list[j]], color='gray')

line, = ax.plot([], [], [], 'b-', label='UAV Path')
point, = ax.plot([], [], [], 'ro', label='UAV Position')
lr1_point, = ax.plot([], [], [], 'o', color='orange', label='LR Obstacle 1')
lr2_point, = ax.plot([], [], [], 'o', color='orange', label='LR Obstacle 2')
ud1_point, = ax.plot([], [], [], 'o', color='purple', label='UD Obstacle 1')
ud2_point, = ax.plot([], [], [], 'o', color='purple', label='UD Obstacle 2')
ax.legend()

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    lr1_point.set_data([], [])
    lr1_point.set_3d_properties([])
    lr2_point.set_data([], [])
    lr2_point.set_3d_properties([])
    ud1_point.set_data([], [])
    ud1_point.set_3d_properties([])
    ud2_point.set_data([], [])
    ud2_point.set_3d_properties([])
    return line, point, lr1_point, lr2_point, ud1_point, ud2_point

def animate(i):
    line.set_data(x[:i+1], y[:i+1])
    line.set_3d_properties(z[:i+1])
    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])
    lr1_point.set_data([lr1_x[i]], [lr1_y[i]])
    lr1_point.set_3d_properties([lr1_z[i]])
    lr2_point.set_data([lr2_x[i]], [lr2_y[i]])
    lr2_point.set_3d_properties([lr2_z[i]])
    ud1_point.set_data([ud1_x[i]], [ud1_y[i]])
    ud1_point.set_3d_properties([ud1_z[i]])
    ud2_point.set_data([ud2_x[i]], [ud2_y[i]])
    ud2_point.set_3d_properties([ud2_z[i]])
    return line, point, lr1_point, lr2_point, ud1_point, ud2_point

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    frames=min(len(x), len(lr1_x), len(lr2_x), len(ud1_x), len(ud2_x)),
    interval=50, blit=True
)

plt.show()
