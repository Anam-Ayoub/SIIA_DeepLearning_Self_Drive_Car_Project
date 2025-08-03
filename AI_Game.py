from ursina import *
import torch
import torch.nn as nn
import math
import json
import csv
import os
from panda3d.core import OrthographicLens, NodePath, Camera as PandaCamera
from ursina import application, scene

# ------------------- 1. Define the MLP Model -------------------
class MLP(nn.Module):
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, 32),        
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ------------------- 2. Load the Trained Model -------------------
model = MLP()
model.load_state_dict(torch.load('mlp_behavior_clone.pth', map_location='cpu'))
model.eval()

# ------------------- 3. Constants -------------------
car_pos = Vec3(-31,1,68.7)
car_rot = Vec3(0,-140,0)
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
REAR_STEER_CUTOFF_SPEED = 30.0
CAMERA_MOVE_SPEED = 20.0
TOP_DOWN_CAM_HEIGHT = 60
TOP_DOWN_CAM_MIN_FOV = 10
TOP_DOWN_CAM_MAX_FOV = 120
TOP_DOWN_CAM_ZOOM_SPEED = 5
MINIMAP_HEIGHT = 50
PATH_HEIGHT = 0.7
MAX_PATH_POINTS = 10
PATH_PREDICTION_TIME = 1.5
PATH_STEP_DT = 0.15
ENGINE_IDLE_VOLUME = 0.3
ENGINE_ACCEL_VOLUME = 0.5
BRAKE_VOLUME = 0.7

# Camera offset constant
camera_offset = Vec3(0, 2, -8)

# ------------------- 4. Initialize App -------------------
app = Ursina(size=(1920, 1080))

# Sound state variables
is_playing_idle = False
is_playing_accel = False
is_braking = False
brake_sound_playing = False
sounds_enabled = False

class CustomEditorCamera(EditorCamera):
    def input(self, key):
        if key in ('w','a','s','d','w hold','a hold','s hold','d hold'):
            return
        super().input(key)

free_cam = CustomEditorCamera(enabled=False)
free_with_car_cam = CustomEditorCamera(enabled=False)

# ------------------- 5. Scene Setup -------------------
city = Entity(model='models/map.glb', position=Vec3(0,0.5,0), scale=1.6, metallic=0.2, smoothness=0.4)
car = Entity(
    model='models/car.glb', 
    position=car_pos, 
    rotation=car_rot,
    metallic=5, 
    smoothness=0.9
)
car.y = 1
car.speed = 0
car.max_speed = CAR_MAX_SPEED

for attr, val in [
    ('acceleration', CAR_ACCELERATION),
    ('deceleration', CAR_DECELERATION),
    ('brake_deceleration', CAR_BRAKE_DECELERATION),
    ('turn_speed', CAR_TURN_SPEED),
    ('turn_smoothness', CAR_TURN_SMOOTHNESS),
    ('turn_amount', 0),
    ('turn_slow_factor', CAR_TURN_SLOW_FACTOR)
]:
    setattr(car, attr, val)

wheel_pos = {
    'front_right': Vec3(0.6, -0.2, 1),
    'front_left': Vec3(-0.6, -0.2, 1),
    'back_right': Vec3(0.6, -0.2, -1.1),
    'back_left': Vec3(-0.6, -0.2, -1.1)
}

Front_right_Wheel = Entity(model='models/wheels/Front_Wheel.glb', parent=car, position=wheel_pos['front_right'])
Front_left_Wheel = Entity(model='models/wheels/Front_Wheel.glb', parent=car, position=wheel_pos['front_left'], rotation=Vec3(0,180,0))
Back_right_Wheel = Entity(model='models/wheels/Back_Wheel.glb', parent=car, position=wheel_pos['back_right'])
Back_left_Wheel = Entity(model='models/wheels/Back_Wheel.glb', parent=car, position=wheel_pos['back_left'], rotation=Vec3(0,180,0))

path_line = Entity(
    model=Mesh(vertices=[], mode='line', thickness=4),
    color=color.yellow,
    enabled=False
)

# Load sound files
try:
    engine_idle_sound = Audio('misc/engine_idle.wav', loop=True, autoplay=False, volume=ENGINE_IDLE_VOLUME)
    engine_accel_sound = Audio('misc/engine_accelerate.wav', loop=True, autoplay=False, volume=ENGINE_ACCEL_VOLUME)
    brake_sound = Audio('misc/brake_screech.wav', loop=False, autoplay=False, volume=BRAKE_VOLUME)
except:
    print("Warning: Could not load sound files")
    engine_idle_sound = None
    engine_accel_sound = None
    brake_sound = None

# ------------------- 6. Minimap Setup -------------------
minimap_dr = application.base.win.make_display_region(0.7, 0.99, 0.7, 0.99)
minimap_dr.set_sort(1)
minimap_dr.set_clear_color_active(True)
minimap_dr.set_clear_color(window.color)
minimap_dr.set_clear_depth_active(True)
minimap_cam_np = NodePath(PandaCamera('minimap'))
minimap_cam_np.reparent_to(scene)
minimap_cam_np.set_hpr(0, -90, 0)
lens = OrthographicLens()
lens.set_film_size(80, 80)
minimap_cam_np.node().set_lens(lens)
minimap_dr.set_camera(minimap_cam_np)
minimap_dr.set_active(False)

# ------------------- 7. Utilities -------------------
def clamp(v, a, b): return max(a, min(v, b))
def distance(p, q): return math.sqrt((p.x - q.x)**2 + (p.y - q.y)**2 + (p.z - q.z)**2)

def point_to_segment_distance(p, a, b):
    ap, ab = p - a, b - a
    sq = ab.x**2 + ab.y**2 + ab.z**2
    if sq == 0: return distance(p, a)
    t = clamp(ap.dot(ab) / sq, 0, 1)
    return distance(p, a + ab * t)

def predict_car_path(car, model, steps, dt):
    positions = []
    current_pos = Vec3(car.position)
    current_rot_y = car.rotation_y
    current_speed = car.speed
    current_turn = car.turn_amount

    state = torch.tensor([[current_pos.x, current_pos.z, current_rot_y]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(state)
    probs = torch.sigmoid(logits).numpy()[0]
    w, a, s, d = (probs > 0.5).astype(int)

    for _ in range(steps):
        forward = w - s
        if forward:
            current_speed += forward * CAR_ACCELERATION * dt
        else:
            if current_speed > 0:
                current_speed -= CAR_DECELERATION * dt
            elif current_speed < 0:
                current_speed += CAR_DECELERATION * dt
        current_speed = clamp(current_speed, -CAR_MAX_SPEED, CAR_MAX_SPEED)

        turn = a - d
        if abs(current_speed) > 0.1 and turn:
            df = 1 if current_speed >= 0 else -1
            current_turn = lerp(current_turn, turn * df, CAR_TURN_SMOOTHNESS * dt)
        else:
            current_turn = lerp(current_turn, 0, CAR_TURN_SMOOTHNESS * dt)

        speed_factor = clamp(abs(current_speed) / CAR_MAX_SPEED, 0.1, 1)
        current_rot_y -= current_turn * CAR_TURN_SPEED * dt * speed_factor * CAR_TURN_SLOW_FACTOR

        forward_vec = Vec3(0, 0, 1)
        forward_vec = Vec3(
            math.sin(math.radians(current_rot_y)) * forward_vec.z,
            0,
            math.cos(math.radians(current_rot_y)) * forward_vec.z
        )
        current_pos += forward_vec * current_speed * dt
        positions.append(Vec3(current_pos.x, PATH_HEIGHT, current_pos.z))

    return positions

def update_sounds(w, a, s, d):
    global is_playing_idle, is_playing_accel, is_braking, brake_sound_playing
    
    if not sounds_enabled or engine_idle_sound is None or engine_accel_sound is None or brake_sound is None:
        return
    
    if car.speed == 0 and not is_playing_idle:
        if is_playing_accel:
            engine_accel_sound.stop()
            is_playing_accel = False
        engine_idle_sound.play()
        is_playing_idle = True
    elif car.speed > 0 and not is_playing_accel and w:
        if is_playing_idle:
            engine_idle_sound.stop()
            is_playing_idle = False
        engine_accel_sound.play()
        is_playing_accel = True
    elif car.speed > 0 and is_playing_accel and not w:
        engine_accel_sound.stop()
        is_playing_accel = False
        engine_idle_sound.play()
        is_playing_idle = True
    
    should_brake = s
    
    if should_brake and not brake_sound_playing and abs(car.speed) > 5:
        brake_sound.play()
        brake_sound_playing = True
    elif not should_brake and brake_sound_playing:
        brake_sound.stop()
        brake_sound_playing = False

def stop_all_sounds():
    global is_playing_idle, is_playing_accel, is_braking, brake_sound_playing
    
    if engine_idle_sound: engine_idle_sound.stop()
    if engine_accel_sound: engine_accel_sound.stop()
    if brake_sound: brake_sound.stop()
    
    is_playing_idle = False
    is_playing_accel = False
    is_braking = False
    brake_sound_playing = False

# ------------------- 8. Load Track -------------------
def load_segments(fn):
    try:
        data = json.load(open(fn))
    except:
        return []
    segs = []
    for pts in data.values():
        vecs = [Vec3(x, car.y, z) for x, z in pts]
        uniq = [vecs[0]] if vecs else []
        for p in vecs[1:]:
            if p != uniq[-1]: uniq.append(p)
        for i in range(len(uniq) - 1):
            segs.append((uniq[i], uniq[i+1]))
    return segs

all_segs = load_segments('coords.json')
for a, b in all_segs:
    Entity(model=Mesh(vertices=[a, b], mode='line', thickness=2), color=color.red)

finish_line = [Vec3(64.35, car.y, 1.30), Vec3(85.28, car.y, 1.30)]
Entity(model=Mesh(vertices=finish_line, mode='line', thickness=4), color=color.yellow)
finish_seg = (finish_line[0], finish_line[1])

# ------------------- 9. Camera Modes -------------------
camera_modes = ['follow', 'free_with_car', 'top_down', 'free', 'ai']
camera_mode_index = 0
camera_mode = camera_modes[camera_mode_index]
ai_active = False

# ------------------- 10. Game Functions -------------------
data_buffer = []
collecting_data = False
data_file = 'data.csv'

def init_csv():
    if not os.path.exists(data_file) or os.stat(data_file).st_size == 0:
        with open(data_file,'a',newline='') as f:
            csv.writer(f).writerow(['x','z','heading','w','a','s','d'])

def restart_game():
    global data_buffer, collecting_data, ai_active, sounds_enabled
    
    car.position, car.rotation = car_pos, car_rot
    car.speed, car.turn_amount = 0, 0
    for wheel in (Front_right_Wheel, Front_left_Wheel, Back_right_Wheel, Back_left_Wheel):
        wheel.rotation = Vec3(0, wheel.rotation_y, 0)
    
    game_over.enabled = victory.enabled = restart_btn.enabled = exit_btn.enabled = False
    data_buffer, collecting_data = [], False
    ai_active = False
    sounds_enabled = False
    set_camera_mode('follow')
    path_line.enabled = False
    stop_all_sounds()

# ------------------- 11. UI -------------------
# Main UI elements
game_over = Text(
    'GAME OVER', 
    scale=3, 
    color=color.red, 
    enabled=False, 
    background=True,
    position=(0, 0.2, -10)
)

victory = Text(
    'YOU WIN!',  
    scale=3, 
    color=color.lime, 
    enabled=False, 
    background=True,
    position=(0, 0.2, -10)
)

mode_text = Text(
    'Follow Mode', 
    position=window.top_left + Vec2(0.1, -0.1),
    scale=1.5, 
    background=True
)

speedometer = Text(
    'Speed: 0 km/h', 
    position=window.bottom_right - Vec2(0.2, 0.1), 
    origin=(1,0), 
    scale=1.5, 
    background=True
)

car_pos_text = Text(
    '', 
    position=window.bottom_left + Vec2(0.1, 0.1),
    scale=1.25, 
    background=True
)

controls = Text(
    'Controls:\nSPACE: Toggle AI drive\nV: Change camera view\nR: Restart game\n\nCamera Modes:\n1. Follow\n2. Free with car\n3. Top-down\n4. Free\n5. AI view', 
    position=window.top_left + Vec2(0.1, -0.25),
    scale=1, 
    background=True,
    line_height=1.2
)

decision_text = Text(
    '', 
    position=(0, -0.45), 
    origin=(0.5,0), 
    scale=1.25, 
    background=True,
    enabled=False
)

# Buttons
restart_btn = Button(
    'Restart', 
    position=(0, -0.2), 
    scale=(0.15,0.1), 
    color=color.azure, 
    highlight_color=color.cyan, 
    enabled=False, 
    on_click=restart_game
)

exit_btn = Button(
    'Exit',    
    position=(0, -0.35), 
    scale=(0.15,0.1), 
    color=color.red, 
    highlight_color=color.orange, 
    enabled=False, 
    on_click=application.quit
)

def set_camera_mode(mode):
    global camera_mode, camera_mode_index
    camera_mode, camera_mode_index = mode, camera_modes.index(mode)
    mode_text.text = mode.replace('_',' ').title() + ' Mode'
    mouse.locked = False
    free_cam.enabled = free_with_car_cam.enabled = False
    minimap_dr.set_active(mode == 'ai')
    decision_text.enabled = (mode == 'ai')
    path_line.enabled = (mode == 'ai')
    
    if mode in ('follow','ai'):
        camera.parent = car
        camera.orthographic = False
        camera.fov = 75
        camera.position = Vec3(0,2,-8)
        camera.rotation = Vec3(10,0,0)
    elif mode == 'free_with_car':
        free_with_car_cam.enabled = True
        free_with_car_cam.position = car.world_position + camera_offset
        free_with_car_cam.rotation = Vec3(10,180,0)
        camera.orthographic = False
        camera.fov = 75
    elif mode == 'top_down':
        camera.parent = car
        camera.orthographic = True
        camera.position = Vec3(0, TOP_DOWN_CAM_HEIGHT - car.y, 0)
        camera.rotation = Vec3(90,0,0)
        camera.fov = TOP_DOWN_CAM_MAX_FOV
    elif mode == 'free':
        free_cam.position, free_cam.rotation = camera.position, camera.rotation
        free_cam.enabled = True
        camera.orthographic = False
        camera.fov = 75

def input(key):
    global ai_active, camera_mode_index, sounds_enabled
    
    if game_over.enabled or victory.enabled:
        if key == 'r':
            restart_game()
        return
    
    if key == 'v':
        set_camera_mode(camera_modes[(camera_mode_index + 1) % len(camera_modes)])
    if camera_mode == 'top_down' and key in ('scroll up','scroll down'):
        delta = -TOP_DOWN_CAM_ZOOM_SPEED if key == 'scroll down' else TOP_DOWN_CAM_ZOOM_SPEED
        camera.fov = clamp(camera.fov + delta, TOP_DOWN_CAM_MIN_FOV, TOP_DOWN_CAM_MAX_FOV)
    if key == 'space' and not ai_active:
        ai_active = True
        sounds_enabled = True

def update():
    global collecting_data, data_buffer
    
    if game_over.enabled or victory.enabled:
        collecting_data = False
        return
    
    if ai_active:
        x, z, h = car.position.x, car.position.z, car.rotation_y
        state = torch.tensor([[x, z, h]], dtype=torch.float32)
        with torch.no_grad():
            logits = model(state)
        probs = torch.sigmoid(logits).numpy()[0]
        w, a, s, d = (probs > 0.5).astype(int)
        
        update_sounds(w, a, s, d)
        
        actions = []
        if w: actions.append('Forward')
        if s: actions.append('Backward')
        if a: actions.append('Left')
        if d: actions.append('Right')
        decision_text.text = (
            f"Actions: {', '.join(actions) or 'Idle'} | "
            f"P(F)={probs[0]:.2f}, P(L)={probs[1]:.2f}, P(B)={probs[2]:.2f}, P(R)={probs[3]:.2f}"
        )
        
        if camera_mode == 'ai':
            steps = int(PATH_PREDICTION_TIME / PATH_STEP_DT)
            predicted_positions = predict_car_path(car, model, steps, PATH_STEP_DT)
            path_line.model = Mesh(
                vertices=predicted_positions[:MAX_PATH_POINTS],
                mode='line',
                thickness=4
            )
            path_line.color = color.yellow
            path_line.enabled = True
        
        if not collecting_data and w: 
            collecting_data = True
        if collecting_data: 
            data_buffer.append([x, z, h, w, a, s, d])
        
        forward = w - s
        if forward:
            car.speed += forward * car.acceleration * time.dt
        else:
            if car.speed > 0:
                car.speed -= car.deceleration * time.dt
            elif car.speed < 0:
                car.speed += car.deceleration * time.dt
        
        car.speed = clamp(car.speed, -car.max_speed, car.max_speed)
        if abs(car.speed) < 0.1:
            car.speed = 0
        
        turn = a - d
        if abs(car.speed) > 0.1 and turn:
            df = 1 if car.speed >= 0 else -1
            car.turn_amount = lerp(car.turn_amount, turn * df, car.turn_smoothness * time.dt)
        else:
            car.turn_amount = lerp(car.turn_amount, 0, car.turn_smoothness * time.dt)
        
        speed_factor = clamp(abs(car.speed) / car.max_speed, 0.1, 1)
        car.rotation_y -= car.turn_amount * car.turn_speed * time.dt * speed_factor * car.turn_slow_factor
        car.position += car.forward * car.speed * time.dt
        
        spin = (car.speed / WHEEL_RADIUS) * (180 / math.pi) * time.dt
        for we in (Front_right_Wheel, Back_right_Wheel):
            we.rotation_x += spin
        for we in (Front_left_Wheel, Back_left_Wheel):
            we.rotation_x -= spin
        
        steer = -car.turn_amount * 30
        Front_right_Wheel.rotation_y = steer
        Front_left_Wheel.rotation_y = steer + 180
        
        rear_factor = 0.5 * clamp((REAR_STEER_CUTOFF_SPEED - abs(car.speed)) / REAR_STEER_CUTOFF_SPEED, 0, 1)
        sign = 1 if car.speed >= 0 else -1
        back = -steer * rear_factor * sign
        Back_right_Wheel.rotation_y = back
        Back_left_Wheel.rotation_y = back + 180
    else:
        decision_text.text = ''
        update_sounds(0, 0, 0, 0)
        path_line.enabled = False
    
    speedometer.text = f"Speed: {abs(car.speed)*3.6:.1f} km/h"
    car_pos_text.text = f"Car Pos: X={car.position.x:.1f}, Y={car.position.y:.1f}, Z={car.position.z:.1f}"
    
    if camera_mode == 'ai':
        minimap_cam_np.set_pos(car.world_position + Vec3(0, MINIMAP_HEIGHT, 0))
    
    for a, b in all_segs:
        if point_to_segment_distance(car.position, a, b) < COLLISION_THRESHOLD:
            game_over.enabled = True
            restart_btn.enabled = exit_btn.enabled = True
            stop_all_sounds()
            print("Game Over! Collision detected.")
            return
    
    if point_to_segment_distance(car.position, *finish_seg) < FINISH_LINE_THRESHOLD:
        victory.enabled = True
        restart_btn.enabled = exit_btn.enabled = True
        stop_all_sounds()
        if data_buffer:
            init_csv()
            with open(data_file,'a',newline='') as f:
                csv.writer(f).writerows(data_buffer)
            print(f"Data saved: {len(data_buffer)} rows.")
        data_buffer = []
        print("You Win! Reached the finish line.")

set_camera_mode('follow')
app.run()

