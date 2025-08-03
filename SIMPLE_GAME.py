from ursina import *
from ursina.shaders import unlit_shader

import math
import random

# =============================================================================
# Application Initialization
# =============================================================================
app = Ursina()

# =============================================================================
# Utility Functions
# =============================================================================
def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# =============================================================================
# Scene Setup - Game Entities
# =============================================================================
# Replace city map with a simple plane
city = Entity(
    model='plane',
    scale=(200, 0.2, 200),
    position=(0, 0, 0),
    color=(0.7, 0.7, 0.7, 1),  # lighter grey with full opacity
    rotation_x=0,
    collider='box',
    texture='white_grid',  # Use grid texture for visible lines
    texture_scale=(25, 25),  # Repeat grid across the plane
    shader=unlit_shader
)

# Replace car with a simple box
car = Entity(
    model='cube',
    color=color.yellow,  # Changed to yellow for visibility
    scale=(1.5, 0.7, 3),
    position=Vec3(0, 1, 0),
    rotation=Vec3(0, 0, 0),
    collider='box',
    texture='white_cube',
    shader=unlit_shader
)

car.speed = 0.0
car.max_speed = 10.0
car.acceleration = 15.0
car.deceleration = 6.0
car.brake_deceleration = 20.0
car.turn_speed = 100.0
car.turn_smoothness = 6.0
car.turn_amount = 0.0
car.turn_slow_factor = 0.8

# Add various cubes to the plane
cube_positions = [
    (random.randint(-90, 90), 1, random.randint(-90, 90)) for _ in range(40)
]
cubes = []
for pos in cube_positions:
    cube = Entity(
        model='cube',
        color=color.random_color(),
        scale=(2, 2, 2),
        position=pos,
        collider='box',
        texture='white_cube',
        shader=unlit_shader
    )
    cubes.append(cube)

# =============================================================================
# UI Elements
# =============================================================================
game_over = Text(text='GAME OVER', origin=(0, 0), scale=3, color=color.red, enabled=False, background=True)
victory = Text(text='YOU WIN!', origin=(0, 0), scale=3, color=color.lime, enabled=False, background=True)
restart_btn = Button(text='RESTART', color=color.white, scale=(0.2,0.1), position=(0, -.3), enabled=False, highlight_color=color.yellow)
car_pos_text = Text(text='', position=(-0.7, -0.45), origin=(0, 0), scale=1.25, background=True)

# Controls display
controls_bg = Entity(model='quad', scale=(0.4,0.1), position=(0,0.4), color=color.rgba(0,0,0,150))
arrow_display = Text(text='', position=(0,0.4), origin=(0,0), scale=2, color=color.white)

# =============================================================================
# Game Logic
# =============================================================================
def restart_game():
    car.position = Vec3(0, 1, 0)
    car.rotation = Vec3(0, 0, 0)
    car.speed = 0.0
    game_over.enabled = False
    victory.enabled = False
    restart_btn.enabled = False

restart_btn.on_click = restart_game

# =============================================================================
# Camera Setup & Control
# =============================================================================
camera_mode = 'free'
free_cam = EditorCamera()
free_cam.position = Vec3(0, 10, -20)
free_cam.rotation = Vec3(30, 0, 0)

def input(key):
    global camera_mode
    # Camera switch logic
    if key == 'v':
        if camera_mode == 'free':
            camera_mode = 'follow'
            free_cam.enabled = False
            camera.parent = car
            camera.position = (0, 6, -15)
            camera.rotation = (20, 0, 0)
            camera.fov = 60  # Wider field of view in follow mode
        else:
            camera_mode = 'free'
            free_cam.enabled = True
            camera.parent = None
            camera.position = free_cam.position
            camera.rotation = free_cam.rotation
            camera.fov = 90  # Reset FOV to default in free mode

# =============================================================================
# Main Update Loop
# =============================================================================
def update():
    if game_over.enabled or victory.enabled:
        return
    # Movement: Arrow keys
    move_speed = 10 * time.dt
    turn_speed = 90 * time.dt
    if held_keys['up arrow']:
        car.position += car.forward * move_speed
    if held_keys['down arrow']:
        car.position -= car.forward * move_speed
    if held_keys['right arrow']:
        car.rotation_y += turn_speed
    if held_keys['left arrow']:
        car.rotation_y -= turn_speed
    car_pos_text.text = f'Position: X={car.position.x:.1f}, Y={car.position.y:.1f}, Z={car.position.z:.1f}'

    # Show which arrow keys are held
    held = []
    if held_keys['up arrow']:
        held.append('UP')
    if held_keys['down arrow']:
        held.append('DOWN')
    if held_keys['left arrow']:
        held.append('LEFT')
    if held_keys['right arrow']:
        held.append('RIGHT')
    arrow_display.text = ' + '.join(held) if held else ''

# =============================================================================
# Run the Application
# =============================================================================
app.run()
