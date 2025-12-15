# create_bitmap.py
import struct

# Image dimensions
width, height = 100, 50

# Colors (B, G, R)
left_color = (0x00, 0x00, 0x70)
right_color = (0x00, 0x00, 0x71)

# Each row in BMP must be padded to a multiple of 4 bytes
row_padding = (4 - (width * 3) % 4) % 4

# Build pixel data (bottom-up, left-to-right)
pixels = bytearray()
for _ in range(height):
    row = bytearray()
    for x in range(width):
        if x < width // 2:
            row += bytes(left_color)
        else:
            row += bytes(right_color)
    row += b'\x00' * row_padding
    pixels = row + pixels  # BMP stores rows bottom-up

# File header sizes
pixel_data_offset = 54
file_size = pixel_data_offset + len(pixels)

# BMP headers
bmp_header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, pixel_data_offset)
dib_header = struct.pack('<IIIHHIIIIII',
                         40, width, height, 1, 24, 0,
                         len(pixels), 2835, 2835, 0, 0)

# Write file
with open('half_half.bmp', 'wb') as f:
    f.write(bmp_header)
    f.write(dib_header)
    f.write(pixels)

print("Created 'half_half.bmp'")
