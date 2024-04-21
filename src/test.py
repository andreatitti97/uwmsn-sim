
import random
import math

def distance_between_points(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def generate_random_points(area, min_distance, max_distance):
    # Extracting area dimensions
    area_width, area_height = area
    
    # Generate random points within the specified area
    points = []
    while len(points) < 4:
        # Generate random coordinates within the area
        x = random.uniform(-area_width/2, area_width/2)
        y = random.uniform(-area_height/2, area_height/2)
        new_point = (x, y)
        
        # Check if new point is far enough from existing points
        if all(distance_between_points(new_point, existing_point) >= min_distance for existing_point in points):
            # Check if new point is not too far from existing points
            if all(distance_between_points(new_point, existing_point) <= max_distance for existing_point in points):
                points.append(new_point)
    
    return points

# Example usage
area = (100, 100)  # Area dimensions (width, height)
min_distance = 10  # Minimum distance between points
max_distance = 50  # Maximum distance between points

random_points = generate_random_points(area, min_distance, max_distance)
print("Generated random points:")
for i, point in enumerate(random_points):
    print(f"Point {i+1}: ({point[0]:.2f}, {point[1]:.2f})")

