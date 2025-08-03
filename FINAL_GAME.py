from ursina import *
import json
import math
import csv
import os

# =============================================================================
# CONSTANTS
# =============================================================================
CAR_MAX_SPEED = 25.0
CAR_ACCELERATION = 15.0
CAR_DECELERATION = 6.0
CAR_BRAKE_DECELERATION = 20.0
CAR_TURN_SPEED = 100.0
CAR_TURN_SMOOTHNESS = 6.0
CAR_TURN_SLOW_FACTOR = 0.8
WHEEL_RADIUS = 0.3
COLLISION_THRESHOLD = 0.8
FINISH_LINE_THRESHOLD = 0.8
# How fast before rear steering is completely off:
REAR_STEER_CUTOFF_SPEED = 30.0

# Camera constants for free_with_car and free modes
CAMERA_MOVE_SPEED = 20.0  # Used for free_with_car and free mode translation
CAMERA_ROTATION_SPEED = 100.0  # Unused since EditorCamera handles rotation

# Camera constants for top_down mode
TOP_DOWN_CAM_HEIGHT = 60  # Height of the camera above the car's Y position in top-down mode
TOP_DOWN_CAM_MIN_FOV = 10  # Minimum zoom level (smaller FOV means more zoomed in)
TOP_DOWN_CAM_MAX_FOV = 120  # Maximum zoom level (larger FOV means more zoomed out)
TOP_DOWN_CAM_ZOOM_SPEED = 5  # How fast mouse wheel zooms

# Camera offset for free_with_car mode
camera_offset = Vec3(0, 2, -8)  # Initial position: 2 units above, 8 units behind car

# Sound constants
ENGINE_IDLE_VOLUME = 0.3
ENGINE_ACCEL_VOLUME = 0.5
BRAKE_VOLUME = 0.7

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================
app = Ursina(size=(1920, 1080))  # Explicit window size to fix win-size warning

# Custom EditorCamera to disable WASD inputs, preventing conflict with car controls
class CustomEditorCamera(EditorCamera):
    def input(self, key):
        if key in ('w', 'a', 's', 'd', 'w hold', 'a hold', 's hold', 'd hold'):
            return
        super().input(key)

# EditorCamera for free mode
free_cam = CustomEditorCamera()
free_cam.position = Vec3(-31, 10, 48.7)  # Position above and behind the car's starting point
free_cam.rotation = Vec3(30, 0, 0)       # Tilted downward to view the car
free_cam.enabled = False                 # Disabled by default

# EditorCamera for free_with_car mode
free_with_car_cam = CustomEditorCamera()
free_with_car_cam.enabled = False        # Disabled by default

# =============================================================================
# SOUND EFFECTS
# =============================================================================
# Load sound files from misc directory
try:
    engine_idle_sound = Audio('misc/engine_idle.wav', loop=True, autoplay=False, volume=ENGINE_IDLE_VOLUME)
    engine_accel_sound = Audio('misc/engine_accelerate.wav', loop=True, autoplay=False, volume=ENGINE_ACCEL_VOLUME)
    brake_sound = Audio('misc/brake_screech.wav', loop=False, autoplay=False, volume=BRAKE_VOLUME)
except:
    print("Warning: Could not load one or more sound files. Game will continue without sound.")
    engine_idle_sound = None
    engine_accel_sound = None
    brake_sound = None

# Sound state variables
is_playing_idle = False
is_playing_accel = False
is_braking = False
brake_sound_playing = False

def update_sounds():
    global is_playing_idle, is_playing_accel, is_braking, brake_sound_playing
    
    if engine_idle_sound is None or engine_accel_sound is None or brake_sound is None:
        return
    
    # Handle engine sounds based on car speed and acceleration
    if car.speed == 0 and not is_playing_idle:
        if is_playing_accel:
            engine_accel_sound.stop()
            is_playing_accel = False
        engine_idle_sound.play()
        is_playing_idle = True
    elif car.speed > 0 and not is_playing_accel and held_keys['w']:
        if is_playing_idle:
            engine_idle_sound.stop()
            is_playing_idle = False
        engine_accel_sound.play()
        is_playing_accel = True
    elif car.speed > 0 and is_playing_accel and not held_keys['w']:
        engine_accel_sound.stop()
        is_playing_accel = False
        engine_idle_sound.play()
        is_playing_idle = True
    
    # Handle brake sound - play when space is held or s is pressed
    should_brake = held_keys['space'] or held_keys['s']
    
    if should_brake and not brake_sound_playing and abs(car.speed) > 5:
        brake_sound.play()
        brake_sound_playing = True
    elif not should_brake and brake_sound_playing:
        # Stop the brake sound immediately if not holding brake key anymore
        brake_sound.stop()
        brake_sound_playing = False

# =============================================================================
# UTILITIES
# =============================================================================
def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# =============================================================================
# SCENE SETUP
# =============================================================================
city = Entity(model='models/map.glb', scale=1.6, position=(0, 0.5, 0), metallic=0.2, smoothness=0.4)

car = Entity(model='models/car.glb', scale=1, position=Vec3(-31,1,68.7), rotation=Vec3(0,-140,0), metallic=5, smoothness=0.9)
# Make sure car is slightly above ground (y=1)
car.y = 1

wheel_positions = {
    'front_right': Vec3(0.6,-0.2,1), 'front_left': Vec3(-0.6,-0.2,1),
    'back_right': Vec3(0.6,-0.2,-1.1), 'back_left': Vec3(-0.6,-0.2,-1.1)
}
Front_right_Wheel = Entity(model='models/wheels/Front_Wheel.glb', parent=car, position=wheel_positions['front_right'])
Front_left_Wheel  = Entity(model='models/wheels/Front_Wheel.glb', parent=car, position=wheel_positions['front_left'], rotation=Vec3(0,180,0))
Back_right_Wheel = Entity(model='models/wheels/Back_Wheel.glb',  parent=car, position=wheel_positions['back_right'])
Back_left_Wheel   = Entity(model='models/wheels/Back_Wheel.glb',  parent=car, position=wheel_positions['back_left'], rotation=Vec3(0,180,0))

# movement properties
car.speed             = 0.0
car.max_speed         = CAR_MAX_SPEED
car.acceleration      = CAR_ACCELERATION
car.deceleration      = CAR_DECELERATION
car.brake_deceleration = CAR_BRAKE_DECELERATION
car.turn_speed        = CAR_TURN_SPEED
car.turn_smoothness   = CAR_TURN_SMOOTHNESS
car.turn_amount       = 0.0
car.turn_slow_factor  = CAR_TURN_SLOW_FACTOR

# =============================================================================
# LOAD TRACK COORDINATES
# =============================================================================
coords_file = 'coords.json'

def load_coords_from_json(filename):
    try:
        with open(filename) as f:
            data = json.load(f)
    except:
        return {}, []
    loaded, segments = {}, []
    for key, pts in data.items():
        # Ensure Y coordinate is consistent, maybe 1 like the car
        vecs = [Vec3(x,car.y,z) for x,z in pts]
        unique = [vecs[0]] if vecs else []
        for p in vecs[1:]:
            if p != unique[-1]: unique.append(p)
        loaded[key] = unique
        for i in range(len(unique)-1):
            segments.append((unique[i], unique[i+1]))
    return loaded, segments

loaded_coords, all_line_segments = load_coords_from_json(coords_file)
for pts in loaded_coords.values():
    if pts:
        # Visualise track lines at car.y height
        Entity(model=Mesh(vertices=pts, mode='line', thickness=2), color=color.red)

# Finish line should also be at the same Y height
finish_line = [Vec3(64.35,car.y,1.30), Vec3(85.28,car.y,1.30)]
Entity(model=Mesh(vertices=finish_line, mode='line', thickness=4), color=color.yellow)
finish_line_segment = (finish_line[0], finish_line[1])

# =============================================================================
# CAMERA MODES
# =============================================================================
camera_modes = ['follow', 'free_with_car', 'top_down', 'free']
camera_mode_index = 0
camera_mode = camera_modes[camera_mode_index]

# =============================================================================
# GAME FUNCTIONS
# =============================================================================
def restart_game():
    global data_buffer, collecting_data, camera_offset, is_playing_idle, is_playing_accel, is_braking, brake_sound_playing
    
    # Stop all sounds
    if engine_idle_sound: engine_idle_sound.stop()
    if engine_accel_sound: engine_accel_sound.stop()
    if brake_sound: brake_sound.stop()
    
    # Reset sound states
    is_playing_idle = False
    is_playing_accel = False
    is_braking = False
    brake_sound_playing = False
    
    car.position, car.rotation = Vec3(-31,1,68.7), Vec3(0,-140,0)
    car.speed, car.turn_amount = 0.0, 0.0
    for wheel in (Front_right_Wheel, Front_left_Wheel, Back_right_Wheel, Back_left_Wheel):
        wheel.rotation = Vec3(0, wheel.rotation_y, 0)
    game_over.enabled = victory.enabled = restart_btn.enabled = exit_btn.enabled = False
    data_buffer, collecting_data = [], False
    camera_offset = Vec3(0, 2, -8)  # Reset to initial offset
    set_camera_mode('follow')

def set_camera_mode(mode):
    global camera_mode, camera_mode_index, camera_offset
    if mode not in camera_modes:
        print(f"Warning: Invalid camera mode '{mode}'.")
        return

    camera_mode = mode
    camera_mode_index = camera_modes.index(mode)
    mode_text.text = f'{mode.replace("_", " ").title()} Mode'

    camera.parent = None
    mouse.locked = False
    free_cam.enabled = False
    free_with_car_cam.enabled = False

    if camera_mode == 'follow':
        camera.parent = car
        camera.orthographic = False
        camera.fov = 75
        camera.position = Vec3(0, 2, -8)
        camera.rotation = Vec3(10, 0, 0)
    elif camera_mode == 'free_with_car':
        free_with_car_cam.enabled = True
        camera.orthographic = False
        camera.fov = 75
        free_with_car_cam.position = car.world_position + camera_offset
        free_with_car_cam.rotation = Vec3(10, 180, 0)  # Initial rotation to face car
    elif camera_mode == 'top_down':
        camera.parent = car
        camera.orthographic = True
        camera.position = Vec3(0, TOP_DOWN_CAM_HEIGHT - car.y, 0)
        camera.rotation = Vec3(90, 0, 0)
        camera.fov = TOP_DOWN_CAM_MAX_FOV
    elif camera_mode == 'free':
        free_cam.position = camera.position
        free_cam.rotation = camera.rotation
        free_cam.enabled = True
        camera.orthographic = False
        camera.fov = 75

def input(key):
    global camera_mode_index

    if key == 'v':
        camera_mode_index = (camera_mode_index + 1) % len(camera_modes)
        set_camera_mode(camera_modes[camera_mode_index])
        return

    if camera_mode == 'top_down':
        if key == 'scroll up':
            camera.fov -= TOP_DOWN_CAM_ZOOM_SPEED
            camera.fov = clamp(camera.fov, TOP_DOWN_CAM_MIN_FOV, TOP_DOWN_CAM_MAX_FOV)
        elif key == 'scroll down':
            camera.fov += TOP_DOWN_CAM_ZOOM_SPEED
            camera.fov = clamp(camera.fov, TOP_DOWN_CAM_MIN_FOV, TOP_DOWN_CAM_MAX_FOV)

    if key == 'r' and (game_over.enabled or victory.enabled):
        restart_game()

def point_to_segment_distance(p, a, b):
    ap, ab = p - a, b - a
    ab_len_sq = ab.x**2 + ab.y**2 + ab.z**2
    if ab_len_sq == 0:
        return distance(p, a)
    t = clamp(ap.dot(ab)/ab_len_sq, 0, 1)
    closest_point = a + ab * t
    return distance(p, closest_point)

# =============================================================================
# UI ELEMENTS
# =============================================================================
# Note: The iCCP warning (incorrect sRGB profile) is due to a PNG asset in models/ or UI textures.
# To fix, re-export assets with correct color profiles using an image editor (e.g., Photoshop, GIMP).
speedometer  = Text(text='Speed: 0 km/h', position=window.bottom_right - Vec2(0.2,0.1), origin=(1,0), scale=1.5, background=True)
game_over    = Text(text='GAME OVER', origin=(0,0), scale=3, color=color.red, enabled=False, background=True)
victory      = Text(text='YOU WIN!', origin=(0,0), scale=3, color=color.lime, enabled=False, background=True)
restart_btn  = Button(text='Restart', position=(-0.1,-0.2), scale=(0.15,0.1), color=color.azure, highlight_color=color.cyan, enabled=False, on_click=restart_game)
exit_btn     = Button(text='Exit', position=(0.1,-0.2), scale=(0.15,0.1), color=color.red, highlight_color=color.orange, enabled=False, on_click=application.quit)
mode_text    = Text(text='Follow Mode', position=(-0.7,0.45), origin=(0,0), scale=1.5, background=True)
controls_text = Text(
    text='Controls:\nWASD: Move car\nV: Change camera mode\nArrow keys: Move camera',
    position=(-0.7,0.3), origin=(-0.5,0.5), scale=1.0, background=True
)
car_pos_text = Text(text='', position=(-0.7,-0.45), origin=(0,0), scale=1.25, background=True)

data_buffer = []
collecting_data = False
data_file_path = 'data.csv'

def initialize_csv_header():
    if not os.path.exists(data_file_path) or os.stat(data_file_path).st_size == 0:
        with open(data_file_path,'a',newline='') as f:
            csv.writer(f).writerow(['x','z','heading','w','a','s','d'])

# =============================================================================
# MAIN UPDATE LOOP
# =============================================================================
def update():
    global collecting_data, data_buffer, camera_mode, camera_offset

    if game_over.enabled or victory.enabled:
        collecting_data = False
        return

    # Update sound effects
    update_sounds()

    # --- Data collection ---
    w, a, s, d = int(held_keys['w']), int(held_keys['a']), int(held_keys['s']), int(held_keys['d'])
    if not collecting_data and w:
        collecting_data = True
    if collecting_data:
        data_buffer.append([car.position.x, car.position.z, car.rotation_y, w, a, s, d])

    # --- Car movement ---
    forward = w - s
    braking = held_keys['space'] or held_keys['s']  # Brake with space or s key
    if forward:
        car.speed += forward * car.acceleration * time.dt
    elif braking:
        car.speed -= car.brake_deceleration * time.dt
    else:
        if car.speed > 0:
            car.speed -= car.deceleration * time.dt
        elif car.speed < 0:
            car.speed += car.deceleration * time.dt
    car.speed = clamp(car.speed, -car.max_speed, car.max_speed)
    if abs(car.speed) < 0.1:
        car.speed = 0.0

    turn = a - d
    if abs(car.speed) > 0.1 and turn:
        dir_factor = 1 if car.speed >= 0 else -1
        car.turn_amount = lerp(car.turn_amount, turn * dir_factor, car.turn_smoothness * time.dt)
    else:
        car.turn_amount = lerp(car.turn_amount, 0, car.turn_smoothness * time.dt)

    speed_factor = clamp(abs(car.speed)/car.max_speed, 0.1, 1)
    car.rotation_y -= car.turn_amount * car.turn_speed * time.dt * speed_factor * car.turn_slow_factor
    car.position   += car.forward * car.speed * time.dt

    # --- Wheels ---
    spin = (car.speed / WHEEL_RADIUS) * (180/math.pi) * time.dt
    for wheel in (Front_right_Wheel, Back_right_Wheel):
        wheel.rotation_x += spin
    for wheel in (Front_left_Wheel, Back_left_Wheel):
        wheel.rotation_x -= spin

    steer = -car.turn_amount * 30
    Front_right_Wheel.rotation_y = steer
    Front_left_Wheel.rotation_y  = steer + 180

    speed_ratio = clamp((REAR_STEER_CUTOFF_SPEED - abs(car.speed)) / REAR_STEER_CUTOFF_SPEED, 0.0, 1.0)
    rear_steer_factor = 0.5 * speed_ratio
    sign = 1 if car.speed >= 0 else -1
    back_mag = -steer * rear_steer_factor * sign
    Back_right_Wheel.rotation_y = back_mag
    Back_left_Wheel.rotation_y  = back_mag + 180

    # --- UI update ---
    car_pos_text.text = f'Car Pos: X={car.position.x:.1f}, Y={car.position.y:.1f}, Z={car.position.z:.1f}'
    speedometer.text    = f'Speed: {abs(car.speed)*3.6:.1f} km/h'

    # --- Camera modes & camera movement ---
    if camera_mode == 'follow':
        camera.parent = car
        camera.orthographic = False
        camera.position = Vec3(0, 2, -8)
        camera.rotation = Vec3(10, 0, 0)
    elif camera_mode == 'free_with_car' and free_with_car_cam.enabled:
        # Update offset with arrow key inputs
        move_speed = CAMERA_MOVE_SPEED * time.dt
        if held_keys['up arrow']:
            camera_offset += free_with_car_cam.forward * move_speed
        if held_keys['down arrow']:
            camera_offset -= free_with_car_cam.forward * move_speed
        if held_keys['left arrow']:
            camera_offset -= free_with_car_cam.right * move_speed
        if held_keys['right arrow']:
            camera_offset += free_with_car_cam.right * move_speed
        if held_keys['q']:
            camera_offset += free_with_car_cam.up * move_speed
        if held_keys['e']:
            camera_offset -= free_with_car_cam.up * move_speed
        # Clamp camera offset to prevent moving too far from car
        camera_offset = Vec3(
            clamp(camera_offset.x, -20, 20),
            clamp(camera_offset.y, 1, 20),
            clamp(camera_offset.z, -30, -5)
        )
        # Update camera position to follow car
        free_with_car_cam.position = car.world_position + camera_offset
    elif camera_mode == 'top_down':
        pass
    elif camera_mode == 'free' and free_cam.enabled:
        # Arrow key movement for ghost mode
        move_speed = CAMERA_MOVE_SPEED * time.dt
        if held_keys['up arrow']:
            free_cam.position += free_cam.forward * move_speed
        if held_keys['down arrow']:
            free_cam.position -= free_cam.forward * move_speed
        if held_keys['left arrow']:
            free_cam.position -= free_cam.right * move_speed
        if held_keys['right arrow']:
            free_cam.position += free_cam.right * move_speed
        if held_keys['q']:
            free_cam.position += free_cam.up * move_speed
        if held_keys['e']:
            free_cam.position -= free_cam.up * move_speed

    # --- Collisions ---
    for a, b in all_line_segments:
        if point_to_segment_distance(car.position, a, b) < COLLISION_THRESHOLD:
            game_over.enabled = True
            restart_btn.enabled = exit_btn.enabled = True
            # Stop all sounds
            if engine_idle_sound: engine_idle_sound.stop()
            if engine_accel_sound: engine_accel_sound.stop()
            if brake_sound: brake_sound.stop()
            print("Game Over! Collision detected.")
            return

    # --- Finish Line ---
    if point_to_segment_distance(car.position, *finish_line_segment) < FINISH_LINE_THRESHOLD:
        victory.enabled = True
        restart_btn.enabled = exit_btn.enabled = True
        # Stop all sounds
        if engine_idle_sound: engine_idle_sound.stop()
        if engine_accel_sound: engine_accel_sound.stop()
        if brake_sound: brake_sound.stop()
        if data_buffer:
            initialize_csv_header()
            with open(data_file_path,'a',newline='') as f:
                csv.writer(f).writerows(data_buffer)
            print(f"Data saved: {len(data_buffer)} rows.")
        data_buffer = []
        print("You Win! Reached the finish line.")

# Initialize the camera mode on startup
set_camera_mode(camera_modes[camera_mode_index])

# RUN
app.run()