import sys
import json
import logging
import numpy as np
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLU import *
import pyrr
import obj_loader as obj

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class Shader:
    """
    Encapsulates shader creation, compilation, and management.
    """

    def __init__(self, vertex_path, fragment_path):
        self.program = self._create_shader(vertex_path, fragment_path)

    def _create_shader(self, vertex_path, fragment_path):
        """
        Compile and link vertex and fragment shaders.

        :param vertex_path: Path to the vertex shader file.
        :param fragment_path: Path to the fragment shader file.
        :return: Compiled shader program.
        """
        try:
            with open(vertex_path, 'r') as f:
                vertex_src = f.read()
            with open(fragment_path, 'r') as f:
                fragment_src = f.read()
        except FileNotFoundError as e:
            logging.error(f"Shader file not found: {e}")
            sys.exit()

        try:
            shader = compileProgram(
                compileShader(vertex_src, GL_VERTEX_SHADER),
                compileShader(fragment_src, GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            logging.error(f"Shader compilation error: {e}")
            sys.exit()

        return shader

    def use(self):
        """Activate the shader program."""
        glUseProgram(self.program)

    def set_uniform_matrix4fv(self, name, matrix):
        """Set a 4x4 matrix uniform."""
        location = glGetUniformLocation(self.program, name)
        glUniformMatrix4fv(location, 1, GL_FALSE, matrix)

    def set_uniform_3f(self, name, value):
        """Set a vec3 uniform."""
        location = glGetUniformLocation(self.program, name)
        glUniform3fv(location, 1, value)

    def destroy(self):
        """Delete the shader program."""
        glDeleteProgram(self.program)


class Camera:
    """
    Encapsulates camera functionality, including position, target, and view matrix.
    """

    def __init__(self, position, target, up):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.update_view_matrix()

    def update_view_matrix(self):
        """Update the view matrix based on camera position, target, and up vector."""
        self.view = pyrr.matrix44.create_look_at(
            eye=self.position,
            target=self.target,
            up=self.up
        )


class Material:
    """
    Encapsulates texture loading and management.
    """

    def __init__(self, texture_path):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Load texture image
        image = pg.image.load(texture_path)
        image_width, image_height = image.get_rect().size
        image_data = pg.image.tostring(image, "RGBA")
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        """Bind the texture."""
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):
        """Delete the texture."""
        glDeleteTextures(1, [self.texture])


class Object3D:
    """
    Represents a 3D object loaded from an OBJ file.
    """

    def __init__(self, filename):
        # Load vertices from OBJ file
        vertices = obj.load_obj(filename)
        scaled_vertices = self._scale_vertices(vertices, 8)

        if not self._has_texture_coords(scaled_vertices):
            scaled_vertices = self._add_texture_coords(scaled_vertices)

        # Convert to numpy array for OpenGL usage
        self.vertices = np.array(scaled_vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices) // 8  # 8 floats per vertex

        # Initialize position and rotation
        self.position = np.array([0, 0, 0], dtype=np.float32)
        self.eulers = np.array([0, 0, 0], dtype=np.float32)  # Rotation in degrees

        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Set up vertex attribute pointers
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))

    def _has_texture_coords(self, vertices):
        """
        Check if the vertices already include texture coordinates.
        """
        # Each vertex has 8 floats: vx, vy, vz, tx, ty, nx, ny, nz
        return len(vertices) % 8 == 0


    def _split_array(self, arr, chunk_size):
        """
        Split an array into chunks of a specified size.

        :param arr: Input array.
        :param chunk_size: Size of each chunk.
        :return: List of chunks.
        """
        return np.array_split(arr, len(arr) // chunk_size)

    def _flatten_vertices(self, vertices):
        """
        Flatten a list of vertices into a single list.

        :param vertices: List of vertices.
        :return: Flattened list.
        """
        return [value for vertex in vertices for value in vertex]

    def _scale_vertices(self, vertices, scale):
        """
        Scale and center the vertices of the object.

        :param vertices: List of vertices.
        :param scale: Scaling factor.
        :return: Scaled and centered vertices.
        """
        # Split the array of vertices (8 floats per vertex)
        sub_matrix = self._split_array(vertices, 8)

        # Extract x, y, z coordinates from each vertex
        x_coords = [v[0] for v in sub_matrix]
        y_coords = [v[1] for v in sub_matrix]
        z_coords = [v[2] for v in sub_matrix]

        # Find the bounding box for scaling
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        z_min, z_max = min(z_coords), max(z_coords)

        # Calculate the width, height, and depth
        width = x_max - x_min
        height = y_max - y_min
        depth = z_max - z_min

        # Find the center of the vertices
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center_z = (z_max + z_min) / 2

        # Transform vertices to center them
        transformed_vertices = []
        for vertex in sub_matrix:
            x, y, z, tx, ty, nx, ny, nz = vertex
            transformed_vertices.append([
                x - center_x,
                y - center_y,
                z - center_z,
                tx, ty, nx, ny, nz
            ])

        # Scale the vertices based on the largest dimension
        largest_dim = max(width, height, depth)
        scale_factor = scale / largest_dim

        # Apply scaling to the transformed vertices
        scaled_vertices = []
        for vertex in transformed_vertices:
            scaled_vertices.append([
                vertex[0] * scale_factor,  # vx
                vertex[1] * scale_factor,  # vy
                vertex[2] * scale_factor,  # vz
                vertex[3], vertex[4],      # tx, ty (texture)
                vertex[5], vertex[6],      # nx, ny (normal)
                vertex[7]                  # nz
            ])

        # Flatten and return the scaled vertices
        return self._flatten_vertices(scaled_vertices)

    def get_bounding_box(self):
        """
        Calculate the bounding box of the object.

        :return: (x_min, x_max, y_min, y_max, z_min, z_max)
        """
        sub_matrix = self._split_array(self.vertices, 8)
        x_coords = [v[0] for v in sub_matrix]
        y_coords = [v[1] for v in sub_matrix]
        z_coords = [v[2] for v in sub_matrix]
        return (
            min(x_coords), max(x_coords),
            min(y_coords), max(y_coords),
            min(z_coords), max(z_coords)
        )

    def get_lowest_y(self):
        """
        Calculate the lowest y-coordinate of the object.
        """
        _, _, y_min, _, _, _ = self.get_bounding_box()
        return y_min

    def place_object_on_base(self, base_height=0.0):
        """
        Place the object on top of the base.
        :param base_height: The height of the base (default is 0.0 for a flat plane).
        """
        # Get the lowest y-coordinate of the object
        y_min = self.get_lowest_y()

        # Adjust the object's position so it sits on top of the base
        self.position[1] = base_height - y_min


    def destroy(self):
        """Clean up the VAO and VBO."""
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])


class BaseObject(Object3D):
    def __init__(self):
        base_vertices = [
            # Positions          # Texture Coords  # Normals
            -5.0, 0.0, -5.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            5.0, 0.0, -5.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            5.0, 0.0, 5.0, 1.0, 1.0, 0.0, 1.0, 0.0,
            -5.0, 0.0, 5.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ]

        self.vertices = np.array(base_vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices) // 8

        self.position = np.array([0, -1, 0], dtype=np.float32)
        self.eulers = np.array([0, 0, 0], dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Set up vertex attribute pointers
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))


class GraphicsEngine:
    """
    Main graphics engine class responsible for initializing the OpenGL context,
    loading shaders, managing the main loop, and rendering the scene.
    """

    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize Pygame and OpenGL
        pg.init()
        self.window_size = tuple(self.config["window_size"])
        self.clock = pg.time.Clock()

        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.window_size, flags=pg.OPENGL | pg.DOUBLEBUF)

        # Set up OpenGL settings
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Load shaders
        self.shader = Shader(
            self.config["shaders"]["vertex"],
            self.config["shaders"]["fragment"]
        )
        self.shader.use()

        # Initialize the 3D object
        self.object = Object3D(self.config["objects"]["obj"])

        # Initialize Base
        self.base = BaseObject()
        self.object.place_object_on_base(base_height=-1)

        # Set up projection matrix
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.window_size[0] / self.window_size[1], 0.1, 100, dtype=np.float32
        )
        self.shader.set_uniform_matrix4fv("projection", self.projection)

        # Set up camera
        self.camera = Camera(
            position=[0, 5, 20],
            target=[0, 0, 0],
            up=[0, 3, 0]
        )
        self.shader.set_uniform_matrix4fv("view", self.camera.view)
        self.shader.set_uniform_3f("cameraPos", self.camera.position)

        # Set up texture
        self.texture = Material(self.config["textures"]["tex"])

        # Start the main loop
        self.run()

    def _check_events(self):
        """
        Handle Pygame events such as quitting the application.
        """
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.object.destroy()
                self.shader.destroy()
                self.base.destroy()
                pg.quit()
                sys.exit()

    def _render(self):
        """
        Render the scene by updating the object's rotation, clearing the frame buffer,
        and drawing the object.
        """

        # Update object rotation
        self.object.eulers[2] += 0.5  # Rotate around the Z-axis
        if self.object.eulers[2] > 360:
            self.object.eulers[2] -= 360

        # Clear frame buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.shader.use()


        # Render the base
        base_transform = pyrr.matrix44.create_from_translation(
            vec=self.base.position,
            dtype=np.float32
        )
        self.shader.set_uniform_matrix4fv("model", base_transform)
        self.texture.use()
        glBindVertexArray(self.base.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, self.base.vertex_count)

        # Calculate model matrix
        model_transform = pyrr.matrix44.create_from_eulers(
            np.radians(self.object.eulers), dtype=np.float32
        )
        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=self.object.position,
                dtype=np.float32
            )
        )

        # Upload model matrix to shader
        self.shader.set_uniform_matrix4fv("model", model_transform)

        # Set light and camera positions for lighting calculations
        light_pos = np.array([75, 75, 75], dtype=np.float32)
        view_pos = self.camera.position
        self.shader.set_uniform_3f("lightPos", light_pos)
        self.shader.set_uniform_3f("viewPos", view_pos)

        # Bind texture and draw the object
        self.texture.use()
        glBindVertexArray(self.object.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.object.vertex_count)

        pg.display.flip()
        self.clock.tick(60)

    def run(self):
        """Main loop of the graphics engine."""
        while True:
            self._check_events()
            self._render()


if __name__ == '__main__':
    engine = GraphicsEngine("config.json")