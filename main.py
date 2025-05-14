#region Imports
import pygame
import sys
import math
import random
import bisect
import pygame.mixer
import mido
import os
import tempfile
import subprocess
import pygame.midi
from tqdm import tqdm 
import pygame.mixer
from pydub import AudioSegment
#endregion


#region Constantes et configuration
# 1080x1920 vertical for high quality export
WIDTH, HEIGHT = 540, 960
FPS = 60
GRAVITY = 0.3
TOTAL_FRAMES = 3660  # ~61 seconds at 60 fps
NOTE_COOLDOWN = 200  # ms mini entre deux notes
NOTE_DURATION = 150  # ms de durée de chaque note
MAX_NOTES = None     # None = on boucle la playlist entière
RECORDING = True  # Set to True to export video frames
TEMP_FRAMES_DIR = tempfile.mkdtemp(prefix='game_frames_')
FRAME_PREFIX = os.path.join(TEMP_FRAMES_DIR, 'frame_')
MIDI_CHANNEL = 0  # For background music
COLLISION_CHANNEL = 1  # For collision effects
#endregion




#region Parsing MIDI en liste de notes
# --- 1) PARSING DU MIDI EN LISTE DE NOTES ---
# On charge seulement les messages note_on dont velocity>0
mid = mido.MidiFile('music.mid')
note_sequence = []
for track in mid.tracks:
    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            note_sequence.append(msg.note)
if not note_sequence:
    raise RuntimeError("Aucune note détectée dans music.mid")
if MAX_NOTES:
    note_sequence = note_sequence[:MAX_NOTES]
#endregion


#region Couleurs
# Colors
def rgb(r, g, b): return (r, g, b)
BLACK = rgb(0, 0, 0)
RED   = rgb(255, 50, 50)
WHITE = rgb(255, 255, 255)
GREEN = rgb(0,255,0)
#endregion


#region Initialisation Pygame/MIDI
pygame.init()
pygame.midi.init()
#endregion


#region Setup MIDI Output
# --- Robust MIDI Output Setup ---
output_id = None
for i in range(pygame.midi.get_count()):
    interf, name, is_input, is_output, opened = pygame.midi.get_device_info(i)
    if is_output and not opened:
        output_id = i
        break

try:
    if output_id is None:
        print("⚠️ No physical MIDI device found. Using virtual MIDI...")
        try:
            midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())
        except Exception:
            print("❌ Install a virtual MIDI driver:")
            print("- Mac: Enable IAC Driver in Audio MIDI Setup")
            print("- Windows: Use loopMIDI")
            print("- Linux: Use aconnect")
            pygame.midi.quit()
            pygame.quit()
            sys.exit(1)
    else:
        midi_out = pygame.midi.Output(output_id)
except Exception as e:
    print(f"❌ MIDI output error: {e}")
    pygame.midi.quit()
    pygame.quit()
    sys.exit(1)
#endregion



#region Classe MIDIPlayer
# --- MIDI Timeline Player ---
class MIDIPlayer:
    def __init__(self, midi_file):
        self.midi_file = mido.MidiFile(midi_file)
        self.start_time = 0
        self.events = []
        # Convert MIDI messages to timed events
        current_time = 0
        for msg in self.midi_file:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                self.events.append((current_time, msg))
        self.events.sort(key=lambda x: x[0])
        self.current_event = 0

    def start(self):
        self.start_time = pygame.time.get_ticks()
        self.current_event = 0

    def update(self):
        if self.current_event >= len(self.events):
            return
        elapsed = (pygame.time.get_ticks() - self.start_time) / 1000
        while self.current_event < len(self.events) and self.events[self.current_event][0] <= elapsed:
            msg = self.events[self.current_event][1]
            midi_out.note_on(msg.note, velocity=msg.velocity, channel=MIDI_CHANNEL)
            # Schedule note off
            pygame.time.set_timer(pygame.USEREVENT + self.current_event, int(msg.time * 1000), 1)
            self.current_event += 1
#endregion


#region Initialisation Audio et MIDIPlayer
# Initialize MIDIPlayer before game loop
midi_player = MIDIPlayer('music.mid')
midi_player.start()
yeah_sound = pygame.mixer.Sound('yeah.mp3')
#endregion




#region Fonctions utilitaires audio
# --- Collision sound effect function ---
def play_collision_sound(event_type):
    # event_type: 'ball-ball' or 'ring'
    if event_type == 'ball-ball':
        midi_out.note_on(60, velocity=100, channel=COLLISION_CHANNEL)  # Low pitch
        pygame.time.set_timer(pygame.USEREVENT + 999, 100, 1)  # Short note

    elif event_type == 'ring':
        midi_out.note_on(72, velocity=100, channel=COLLISION_CHANNEL)  # Higher pitch
        pygame.time.set_timer(pygame.USEREVENT + 998, 100, 1)  # Short note

    elif event_type == 'yeah':
        yeah_sound.stop()  # Cut previous sound if playing
        yeah_sound.play()
#endregion


# Hide window for fast export (headless mode)
#region Initialisation écran, polices, variables de score
os.environ['SDL_VIDEODRIVER'] = 'dummy'
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
# Font for score
font = pygame.font.SysFont('Arial', 48)

# Scores
score_red = 0
score_blue = 0

passed_indices = []
#endregion


#region Classe Ball
class Ball:
    def __init__(self, x, y, radius, speed=6):
        self.pos = pygame.Vector2(x, y)
        theta = random.uniform(0, 2 * math.pi)
        self.vel = pygame.Vector2(math.cos(theta), math.sin(theta)) * speed
        self.radius = radius
        self.trail = []  # liste des positions précédentes
        self.trail_max_length = 20  # nombre d'éléments dans la traînée

    def update(self):
        self.vel.y += GRAVITY
        self.pos += self.vel
        # Ajout de la position actuelle à la traînée
        self.trail.append(self.pos.copy())
        if len(self.trail) > self.trail_max_length:
            self.trail.pop(0)

    def draw(self, surface, color=RED):
        # Dessine la traînée avec un dégradé et de la transparence
        for i, p in enumerate(self.trail):
            alpha = int(255 * (i + 1) / len(self.trail)) if len(self.trail) > 1 else 255
            trail_radius = int(self.radius * (0.6 + 0.4 * (i + 1) / len(self.trail)))
            if color == RED:
                trail_color = (255, 120, 120)
            else:
                trail_color = (120, 180, 255)
            trail_surf = pygame.Surface((trail_radius*2, trail_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, trail_color + (alpha//2,), (trail_radius, trail_radius), trail_radius)
            surface.blit(trail_surf, (p.x - trail_radius, p.y - trail_radius), special_flags=pygame.BLEND_PREMULTIPLIED)
        # Dessine la balle
        pygame.draw.circle(surface, WHITE, self.pos, self.radius + 3)  # outline
        pygame.draw.circle(surface, color,   self.pos, self.radius)

    def try_bounce_or_pass(self, center, ring):
        """
        ring: dict with keys radius, start, end
        Returns:
            'bounce' if bounced,
            'pass' if goes through gap,
             None otherwise
        """
        to_center = self.pos - center
        dist = to_center.length()
        arc_width = 4  # largeur de l'arc, comme dans pygame.draw.arc
        min_dist = ring['radius'] - self.radius - arc_width/2
        max_dist = ring['radius'] + self.radius + arc_width/2
        if min_dist <= dist <= max_dist:
            # Nouvelle logique d'angle en degrés, sens trigo classique
            vec = self.pos - center
            angle = math.degrees(math.atan2(-vec[1], vec[0])) % 360
            start_angle = math.degrees(ring['start']) % 360
            end_angle = math.degrees(ring['end']) % 360
            inside_arc = False
            if start_angle < end_angle:
                inside_arc = (angle > start_angle) and (angle < end_angle)
            else:
                inside_arc = (angle > start_angle) or (angle < end_angle)

            if inside_arc:
                n = to_center.normalize()
                v_dot_n = self.vel.dot(n)
                v_normal = n * v_dot_n
                v_tangent = self.vel - v_normal
                # Clean reflection: invert normal, keep tangent, apply slight damping
                restitution = 1  # 1.0 = perfectly elastic, <1.0 = some energy loss
                self.vel = (-v_normal * restitution) + v_tangent
                # Move the ball just outside the ring to avoid sticking
                overlap = (dist + self.radius) - ring['radius']
                self.pos -= n * overlap
                return 'bounce'
            else:
                return 'pass'
        return None
#endregion



#region Initialisation des objets de jeu
# Initialize balls in the center and slightly offset
ball = Ball(WIDTH//2, HEIGHT//2, radius=20)
ball2 = Ball(WIDTH//2 + 60, HEIGHT//2, radius=20)

# Infinite effect: 1000 rings, zoom, only visible rings drawn
ring_center = pygame.Vector2(WIDTH//2, HEIGHT//2)
NUM_TOTAL_RINGS = 1000
base_radius = 150
radius_step = 18  # rapproché
rings = []
gap_angle = math.radians(60)
spiral_step = math.radians(5)

# Shrinking animation parameters
shrink_timer = 0    # Timer to track shrinking animation
shrink_speed = 0.08 # Lower value for smooth shrink animation

# Shockwave effect parameters
shockwaves = []  # List of active shockwaves

for i in range(NUM_TOTAL_RINGS):
    # Stocke uniquement l'angle de départ (le rayon sera recalculé dynamiquement)
    start = (i * spiral_step) % (2 * math.pi)
    arc_length = 2 * math.pi - gap_angle
    end = start + arc_length
    rings.append({
        'start': start,
        'end': end,
        'initial_index': i,            # Store the original index for proper spacing
        'original_radius': base_radius + i * radius_step,  # Store the original radius
        'offset': 0,                   # Current radius offset for shrinking
        'target_offset': 0,            # Target radius offset for smooth animation
        'active': True                 # Flag to track if ring is active
    })
#endregion

#region Boucle principale d'animation
# Main animation loop
frame = 0
while frame < TOTAL_FRAMES:
    midi_player.update()  # Play background music according to MIDI timeline
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Handle note_off for MIDIPlayer scheduled events
        if event.type >= pygame.USEREVENT and event.type < pygame.USEREVENT + 900:
            event_idx = event.type - pygame.USEREVENT
            if event_idx < len(midi_player.events):
                msg = midi_player.events[event_idx][1]
                midi_out.note_off(msg.note, velocity=0, channel=MIDI_CHANNEL)
        # Handle note_off for collision effects
        if event.type == pygame.USEREVENT + 998:
            midi_out.note_off(72, velocity=0, channel=COLLISION_CHANNEL)
        if event.type == pygame.USEREVENT + 999:
            midi_out.note_off(60, velocity=0, channel=COLLISION_CHANNEL)

    screen.fill(BLACK)

    # Process ring shrinking animation
    if shrink_timer > 0:
        shrink_timer -= 1
    
    # Apply gradual shrinking/expanding to all active rings
    for ring in rings:
        if ring['active'] and abs(ring['offset'] - ring['target_offset']) > 0.1:
            ring['offset'] += (ring['target_offset'] - ring['offset']) * shrink_speed
            if abs(ring['offset'] - ring['target_offset']) < 0.1:
                ring['offset'] = ring['target_offset']

    # Determine visible rings
    visible_rings = []
    min_visible_radius = 30
    max_visible_radius = max(WIDTH, HEIGHT) / 2 + 100

    for i, ring in enumerate(rings):
        if not ring['active']:
            continue

        # Calculate actual radius with offset
        actual_radius = ring['original_radius'] - ring['offset']
        if min_visible_radius < actual_radius < max_visible_radius:
            visible_rings.append((i, ring))

    # Update & draw visible rings, handle passes & bounces
    delta = math.radians(1)
    rings_passed = False  # Flag to track if any rings were passed this frame
    passed_ring_index = -1  # Index of the passed ring

    for i, ring in visible_rings:
            # Update the rotation angle
            ring['start'] += delta
            ring['end'] += delta

            # Calculate actual radius with offset
            actual_radius = ring['original_radius'] - ring['offset']

            # Draw ring using the actual radius
            rect = pygame.Rect(
                ring_center.x - actual_radius,
                ring_center.y - actual_radius,
                actual_radius * 2,
                actual_radius * 2
            )
            pygame.draw.arc(screen, WHITE, rect, ring['start'], ring['end'], 4)

            # Collision test for red ball
            ring_for_collision = ring.copy()
            ring_for_collision['radius'] = actual_radius
            result = ball.try_bounce_or_pass(ring_center, ring_for_collision)
            if result == 'pass':
                ring['active'] = False
                score_red += 1
                rings_passed = True
                # Add to passed_indices (sorted)
                idx = ring['initial_index']
                if idx not in passed_indices:  # Prevent duplicates
                    bisect.insort(passed_indices, idx)
                # Add shockwave effect for red ball
                shockwaves.append({
                    'center': ring_center.copy(),
                    'radius': actual_radius,
                    'max_radius': actual_radius + 300,
                    'alpha': 255,
                    'color': RED
                })

            # Collision test for blue ball
            ring_for_collision2 = ring.copy()
            ring_for_collision2['radius'] = actual_radius
            result2 = ball2.try_bounce_or_pass(ring_center, ring_for_collision2)
            if result2 == 'pass':
                ring['active'] = False
                score_blue += 1
                rings_passed = True
                # Add to passed_indices (sorted)
                idx = ring['initial_index']
                if idx not in passed_indices:  # Prevent duplicates
                    bisect.insort(passed_indices, idx)
                # Add shockwave effect for blue ball
                shockwaves.append({
                    'center': ring_center.copy(),
                    'radius': actual_radius,
                    'max_radius': actual_radius + 300,
                    'alpha': 255,
                    'color': GREEN
                })

    # Update target offsets for all rings based on passed_indices
    if rings_passed:
        for r in rings:
            if r['active']:
                # Number of passed rings with index < current ring's index
                count_passed_before = bisect.bisect_left(passed_indices, r['initial_index'])
                r['target_offset'] = count_passed_before * radius_step
        # Start shrinking animation
        shrink_timer = 60  # 1 second at 60 FPS
    # Update ball physics
    ball.update()
    ball2.update()

    # Ball-ball collision (elastic, like tennis balls)
    diff = ball2.pos - ball.pos
    dist = diff.length()
    min_dist = ball.radius + ball2.radius
    if dist < min_dist and dist > 0:
        n = diff.normalize()
        # Project velocities onto the normal
        v1n = ball.vel.dot(n)
        v2n = ball2.vel.dot(n)
        # Exchange normal components
        ball.vel += n * (v2n - v1n)
        ball2.vel += n * (v1n - v2n)
        # Separate balls so they don't overlap
        overlap = min_dist - dist
        ball.pos -= n * (overlap / 2)
        ball2.pos += n * (overlap / 2)

    # Draw shockwave effects
    for sw in shockwaves[:]:
        sw['radius'] += 12  # Speed of expansion
        sw['alpha'] = max(0, sw['alpha'] - 12)  # Fade out
        if sw['radius'] > sw['max_radius'] or sw['alpha'] <= 0:
            shockwaves.remove(sw)
            continue
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(
            surf,
            sw['color'] + (int(sw['alpha']),),
            (int(sw['center'].x), int(sw['center'].y)),
            int(sw['radius']),
            8
        )
        screen.blit(surf, (0, 0))

    # Draw both balls
    ball.draw(screen, RED)
    ball2.draw(screen, GREEN)

    # --- UI Overlay: Title, Score Table, Clock ---
    # Fonts
    title_font = pygame.font.SysFont('Arial', 32, bold=True)
    score_font = pygame.font.SysFont('Arial', 24, bold=True)
    clock_font = pygame.font.SysFont('Arial', 26, bold=True)

    # Title
    title_text = "Will your crush love you ?"
    title_surf = title_font.render(title_text, True, (0,0,0))
    title_bg_rect = pygame.Rect(WIDTH//2 - title_surf.get_width()//2 - 12, 16, title_surf.get_width() + 24, title_surf.get_height() + 8)
    pygame.draw.rect(screen, (255,255,255), title_bg_rect, border_radius=8)
    screen.blit(title_surf, (title_bg_rect.x + 12, title_bg_rect.y + 4))

    # Score Table - two independent boxes
    yes_text = f"Yes : {score_red}"
    no_text = f"No : {score_blue}"
    yes_surf = score_font.render(yes_text, True, (0,255,0))
    no_surf = score_font.render(no_text, True, (255,60,60))
    box_padding = 12
    box_gap = 30
    box_y = title_bg_rect.bottom + 8
    yes_bg_rect = pygame.Rect(WIDTH//2 - yes_surf.get_width() - box_gap//2 - box_padding, box_y, yes_surf.get_width() + 2*box_padding, yes_surf.get_height() + 8)
    no_bg_rect = pygame.Rect(WIDTH//2 + box_gap//2, box_y, no_surf.get_width() + 2*box_padding, no_surf.get_height() + 8)
    pygame.draw.rect(screen, (255,255,255), yes_bg_rect, border_radius=8)
    pygame.draw.rect(screen, (255,255,255), no_bg_rect, border_radius=8)
    pygame.draw.rect(screen, (0,255,0), yes_bg_rect, 2, border_radius=8)
    pygame.draw.rect(screen, (255,60,60), no_bg_rect, 2, border_radius=8)
    screen.blit(yes_surf, (yes_bg_rect.x + box_padding, yes_bg_rect.y + 6))
    screen.blit(no_surf, (no_bg_rect.x + box_padding, no_bg_rect.y + 6))

    # Clock (bottom center, lifted higher)
    seconds_left = max(0, (TOTAL_FRAMES - frame)//FPS)
    mins = seconds_left // 60
    secs = seconds_left % 60
    clock_str = f"{mins:02}:{secs:02}"
    clock_surf = clock_font.render(clock_str, True, (0,0,0))
    clock_bg_rect = pygame.Rect(WIDTH//2 - clock_surf.get_width()//2 - 14, HEIGHT - clock_surf.get_height() - 80, clock_surf.get_width() + 28, clock_surf.get_height() + 12)
    pygame.draw.rect(screen, (255,255,255), clock_bg_rect, border_radius=7)
    screen.blit(clock_surf, (clock_bg_rect.x + 14, clock_bg_rect.y + 6))

    # --- Export frame as PNG if enabled ---
    if RECORDING:
        pygame.image.save(screen, f"{FRAME_PREFIX}{frame:05d}.png")
        # Print a progress bar in the terminal every 60 frames
        if frame % 60 == 0 or frame == TOTAL_FRAMES - 1:
            percent = int(100 * frame / TOTAL_FRAMES)
            bar_len = 40
            filled_len = int(bar_len * percent // 100)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            print(f"\r[Export] |{bar}| {percent}% ({frame}/{TOTAL_FRAMES})", end='', flush=True)
    # Skip display.flip() and clock.tick(FPS) for speed
    #clock.tick(FPS)
    frame += 1
    #actual_fps = clock.get_fps()
    #if actual_fps > FPS * 1.1:  # If running too fast
    #    time.sleep(0.001)  # Prevent 100% CPU usage
#endregion

#region Nettoyage et post-processing
# Nettoyage
# Cleanup all notes before quitting
for ch in range(16):
    try:
        midi_out.note_off_all(channel=ch)
    except Exception:
        pass
midi_out.close()
pygame.midi.quit()
pygame.quit()

# Updated video compression command
def compress_frames_to_video():
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # overwrite without asking
        '-framerate', str(FPS),
        '-i', f"{FRAME_PREFIX}%05d.png",
        '-vf', "scale=1080:1920:force_original_aspect_ratio=1,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        '-c:v', 'libx265',  # HEVC codec for better compression
        '-crf', '28',  # Quality level (18-28 is good, lower=better quality)
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        'game_recording_compressed.mp4'
    ]
    print("\n[FFmpeg] Compressing frames into video... (this may take a minute)")
    subprocess.run(ffmpeg_cmd)
    # Clean up temporary frames
    for f in os.listdir(TEMP_FRAMES_DIR):
        os.remove(os.path.join(TEMP_FRAMES_DIR, f))
    os.rmdir(TEMP_FRAMES_DIR)
    print("[FFmpeg] Done! Output: game_recording_compressed.mp4")

if RECORDING:
    compress_frames_to_video()
#endregion