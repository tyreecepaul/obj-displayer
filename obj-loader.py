# Below code was generated using ChatGPT on 4/01/2025
# Code was altered from Week 1 Computer Graphics Practical

# Returns data in format [vx, vy, vz, tx, ty, nx, ny, nz]

import os

def load_obj(filename):
    vertices = []
    tex_coords = []
    normals = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':  # Vertex position
                vertices.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'vt':  # Texture coordinate
                tex_coords.append(tuple(map(float, parts[1:3])))
            elif parts[0] == 'vn':  # Normal vector
                normals.append(tuple(map(float, parts[1:4])))
            elif parts[0] == 'f':  # Face
                face = []
                for vert in parts[1:]:
                    indices = list(map(lambda x: int(x) - 1 if x else None, vert.split('/')))
                    face.append(indices)
                faces.append(face)

    final_data = []
    for face in faces:
        for vert in face:
            v_idx = vert[0]
            t_idx = vert[1] if len(vert) > 1 and vert[1] is not None else 0
            n_idx = vert[2] if len(vert) > 2 and vert[2] is not None else 0

            v = vertices[v_idx] if v_idx is not None else (0.0, 0.0, 0.0)
            t = tex_coords[t_idx] if t_idx is not None and t_idx < len(tex_coords) else (0.0, 0.0)
            n = normals[n_idx] if n_idx is not None and n_idx < len(normals) else (0.0, 0.0, 0.0)

            final_data.extend([v[0], v[1], v[2], t[0], t[1], n[0], n[1], n[2]])

    return final_data

# Example Usage
if __name__ == "__main__":
    obj_file = "objects/1.obj"
    if os.path.exists(obj_file):
        data = load_obj(obj_file)
        print("Loaded OBJ Data:", data)
    else:
        print("File not found:", obj_file)
