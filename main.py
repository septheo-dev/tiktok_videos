import pygame
import sys
import math
import random

# Reduced window size for better Mac performance
WIDTH, HEIGHT = 540, 960
FPS = 60
GRAVITY = 0.3
TOTAL_FRAMES = 3660  # ~61 seconds at 60 fps

# Colors
def rgb(r, g, b): return (r, g, b)
BLACK = rgb(0, 0, 0)
RED   = rgb(255, 50, 50)
WHITE = rgb(255, 255, 255)
BLUE = rgb(80, 120, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
# Font for score
font = pygame.font.SysFont('Arial', 48)

# Scores
score_red = 0
score_blue = 0

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
                # Reflect normal, strongly dampen tangent
                self.vel = -v_normal + v_tangent * 0.05
                # Si la vitesse devient trop faible, on remet une vitesse normale modérée ET on ajoute un petit kick tangent
                if self.vel.length() < 1.8:
                    self.vel = n * 2.8
                    tangent = pygame.Vector2(-n.y, n.x)
                    self.vel += tangent * (random.uniform(-1, 1) * 0.7)  # kick tangent pour éviter tout blocage
                # Si la composante tangentielle est trop faible, on ajoute un kick horizontal pour éviter les rebonds verticaux à l'infini
                tangent = pygame.Vector2(-n.y, n.x)
                tangential_speed = self.vel.dot(tangent)
                if abs(tangential_speed) < 0.5:
                    self.vel += tangent * (random.choice([-1, 1]) * 1.2)
                overlap = (dist + self.radius) - ring['radius']
                self.pos -= n * overlap
                return 'bounce'
            else:
                return 'pass'
        return None

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
shrink_speed = 0.08  # Lower value for smooth shrink animation

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
            passed_ring_index = i

        # Collision test for blue ball
        ring_for_collision2 = ring.copy()
        ring_for_collision2['radius'] = actual_radius
        result2 = ball2.try_bounce_or_pass(ring_center, ring_for_collision2)
        if result2 == 'pass':
            ring['active'] = False
            score_blue += 1
            rings_passed = True
            passed_ring_index = i

    # If a ring was passed, adjust the target offsets for all active rings
    if rings_passed and passed_ring_index >= 0:
        # Get the radius step amount we need to shrink by
        shrink_amount = radius_step
        
        # Set target offsets for all rings with larger indices
        for r in rings:
            if r['active'] and r['initial_index'] > passed_ring_index:
                r['target_offset'] += shrink_amount
        
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

    # Draw both balls
    ball.draw(screen, RED)
    ball2.draw(screen, BLUE)

    # Draw scores
    score_red_surf = font.render(f"{score_red}", True, RED)
    score_blue_surf = font.render(f"{score_blue}", True, BLUE)
    screen.blit(score_red_surf, (40, 20))
    screen.blit(score_blue_surf, (WIDTH - 40 - score_blue_surf.get_width(), 20))

    pygame.display.flip()
    clock.tick(FPS)
    frame += 1

pygame.quit()