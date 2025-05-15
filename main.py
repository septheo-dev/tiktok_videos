#region Imports
import pygame
import sys
import math
import random
import bisect
import pygame.mixer
import os
import tempfile
import subprocess
from tqdm import tqdm 
import pygame.mixer
from pydub import AudioSegment
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






#region Couleurs
# Colors
def rgb(r, g, b): return (r, g, b)
BLACK = rgb(0, 0, 0)
RED   = rgb(255, 50, 50)
WHITE = rgb(255, 255, 255)
GREEN = rgb(0,255,0)
#endregion


pygame.init()

# Hide window for fast export (headless mode)
#region Initialisation écran, polices, variables de score
os.environ['SDL_VIDEODRIVER'] = 'dummy'
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
yeah_events = []
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
                trail_color = (0,255,0)
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
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

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
                yeah_events.append(frame)
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
                yeah_events.append(frame)
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
    score_font_big = pygame.font.SysFont('Arial', 38, bold=True)
    yes_surf = score_font_big.render(yes_text, True, (255,60,60))  # Yes devient rouge
    no_surf = score_font_big.render(no_text, True, (0,255,0))      # No devient vert
    box_padding = 38
    box_gap = 60
    box_y = title_bg_rect.bottom + 18
    yes_bg_rect = pygame.Rect(WIDTH//2 - yes_surf.get_width() - box_gap//2 - box_padding, box_y, yes_surf.get_width() + 2*box_padding, yes_surf.get_height() + 38)
    no_bg_rect = pygame.Rect(WIDTH//2 + box_gap//2, box_y, no_surf.get_width() + 2*box_padding, no_surf.get_height() + 38)
    pygame.draw.rect(screen, (255,255,255), yes_bg_rect, border_radius=18)
    pygame.draw.rect(screen, (255,255,255), no_bg_rect, border_radius=18)
    pygame.draw.rect(screen, (255,60,60), yes_bg_rect, 4, border_radius=18)  # Yes bordure rouge
    pygame.draw.rect(screen, (0,255,0), no_bg_rect, 4, border_radius=18)     # No bordure verte
    screen.blit(yes_surf, (yes_bg_rect.x + box_padding, yes_bg_rect.y + 18))
    screen.blit(no_surf, (no_bg_rect.x + box_padding, no_bg_rect.y + 18))

    # Clock (bottom center, lifted higher)
    seconds_left = max(0, (TOTAL_FRAMES - frame)//FPS)
    mins = seconds_left // 60
    secs = seconds_left % 60
    clock_str = f"{mins:02}:{secs:02}"
    clock_surf = clock_font.render(clock_str, True, (0,0,0))
    clock_bg_rect = pygame.Rect(WIDTH//2 - clock_surf.get_width()//2 - 24, HEIGHT - clock_surf.get_height() - 120, clock_surf.get_width() + 48, clock_surf.get_height() + 28)
    pygame.draw.rect(screen, (255,255,255), clock_bg_rect, border_radius=14)
    screen.blit(clock_surf, (clock_bg_rect.x + 24, clock_bg_rect.y + 14))

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
    frame += 1

pygame.quit()
#endregion



#create text audio yeah file

def create_next_output_folder():
    """
    Crée un dossier 'output/video_N' où N est le prochain numéro disponible.
    Retourne le chemin du dossier créé.
    """
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n = 1
    while True:
        candidate = os.path.join(output_dir, f"video_{n}")
        if not os.path.exists(candidate):
            os.makedirs(candidate)
            return candidate
        n += 1

# --- Choix du dossier de sortie ---
output_folder = create_next_output_folder()


def create_audio(yeah_events, output_folder):
    yeah_sound = AudioSegment.from_file("output/yeah.mp3")
    FPS = 60  # Frames per second
    frame_duration_ms = 1000 / FPS  # Duration of one frame in ms
    total_duration_ms = (max(yeah_events) + 1) * frame_duration_ms
    output_audio = AudioSegment.silent(duration=total_duration_ms)
    for frame in yeah_events:
        time_ms = frame * frame_duration_ms
        output_audio = output_audio.overlay(yeah_sound, position=int(time_ms))
    wav_path = os.path.join(output_folder, "output.wav")
    output_audio.export(wav_path, format="wav")
    print(f"{wav_path} created successfully!")

create_audio(yeah_events, output_folder)

# Updated video compression command
def compress_frames_to_video(output_folder):
    mp4_path = os.path.join(output_folder, "game_recording.mp4")
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
        mp4_path
    ]
    print("\n[FFmpeg] Compressing frames into video... (this may take a minute)")
    subprocess.run(ffmpeg_cmd)
    for f in os.listdir(TEMP_FRAMES_DIR):
        os.remove(os.path.join(TEMP_FRAMES_DIR, f))
    os.rmdir(TEMP_FRAMES_DIR)
    print(f"[FFmpeg] Done! Output: {mp4_path}")

if RECORDING:
    compress_frames_to_video(output_folder)


def compile_audio_video(output_folder):
    wav_path = os.path.join(output_folder, "output.wav")
    mp4_path = os.path.join(output_folder, "game_recording.mp4")
    final_path = os.path.join(output_folder, "game_recording_final.mp4")
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # overwrite without asking
        '-i', wav_path,
        '-i', mp4_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:a',
        '-map', '1:v',
        final_path
    ]
    print("\n[FFmpeg] Compiling audio and video... (this may take a minute)")
    subprocess.run(ffmpeg_cmd)
    print(f"[FFmpeg] Done! Output: {final_path}")

compile_audio_video(output_folder)
#endregion