import os

folder = 'export_frames'
files = sorted(f for f in os.listdir(folder) if f.startswith('frame_') and f.endswith('.png'))
for idx, fname in enumerate(files):
    new_name = f"frame_{idx:05d}.png"
    os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))