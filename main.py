import sys
import numpy as np
import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.GLU import *
import pyrr
import obj_loader as obj

class GraphicsEngine:
    """
    Main graphics engine class responsible for initializing the OpenGL context,
    loading shaders, managing the main loop, and rendering the scene.
    """

    def __init__(self, filename, window_size=(1600, 900)):
        """
        Initialize the graphics engine, set up the OpenGL context, and load shaders.

        :param filename: Name of the OBJ file (without extension) to load.
        :param window_size: Tuple representing the window width and height.
        """
        pg.init()
        self.window_size = window_size

        # Set OpenGL attributes and create the window
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode(self.window_size, flags=pg.OPENGL | pg.DOUBLEBUF)

        # Set up OpenGL settings
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        self.clock = pg.time.Clock()

        # Load shaders
        self.shader = self._create_shader()
        glUseProgram(self.shader)

        # Initialize the 3D object
        self.object = Object3D(filename)

        # Set up projection matrix
        self.projection = pyrr.matrix44.create_perspective_projection_matrix(
            45, self.window_size[0] / self.window_size[1], 0.1, 100, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, self.projection
        )

        # Set up view matrix (camera)
        self.view = pyrr.matrix44.create_look_at(
            eye=np.array([0, 0, 60], dtype=np.float32),  # Camera position
            target=np.array([0, 0, 0], dtype=np.float32),  # Look at origin
            up=np.array([0, 1, 0], dtype=np.float32)  # Up vector
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "view"),
            1, GL_FALSE, self.view
        )

        # Get model matrix location
        self.model_matrix_location = glGetUniformLocation(self.shader, "model")

        # Start the main loop
        self.run()

    def _create_shader(self):
        """
        Compile and link the vertex and fragment shaders.

        :return: Compiled shader program.
        """
        with open('shaders/vertex.txt', 'r') as f:
            vertex_src = f.read()
        with open('shaders/fragment.txt', 'r') as f:
            fragment_src = f.read()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )
        return shader

    def _check_events(self):
        """
        Handle Pygame events such as quitting the application.
        """
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.object.destroy()
                pg.quit()
                sys.exit()

    def _render(self):
        """
        Render the scene by updating the object's rotation, clearing the frame buffer,
        and drawing the object.
        """
        # Update object rotation
        self.object.eulers[2] += 0.2
        if self.object.eulers[2] > 360:
            self.object.eulers[2] -= 360

        # Clear frame buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

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
        glUniformMatrix4fv(self.model_matrix_location, 1, GL_FALSE, model_transform)

        # Draw the object
        glBindVertexArray(self.object.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.object.vertex_count)

        # Swap buffers
        pg.display.flip()

    def run(self):
        """
        Main loop of the graphics engine.
        """
        while True:
            self._check_events()
            self._render()
            self.clock.tick(60)


class Object3D:
    """
    Represents a 3D object loaded from an OBJ file.
    """

    def __init__(self, filename):
        """
        Initialize the 3D object by loading vertices from an OBJ file,
        scaling them, and setting up OpenGL buffers.

        :param filename: Name of the OBJ file (without extension).
        """
        # Load vertices from OBJ file
        vertices = obj.load_obj(f"objects/{filename}.obj")
        # Scale and flatten vertices
        scaled_vertices = self._scale_vertices(vertices, 8)

        # Convert to numpy array for OpenGL usage
        self.vertices = np.array(scaled_vertices, dtype=np.float32)
        self.vertex_count = len(self.vertices) // 8  # 8 floats per vertex

        # Initialize position and rotation
        self.position = np.array([0, 0, 50], dtype=np.float32)
        self.eulers = np.array([0, 0, 0], dtype=np.float32)  # Rotation in degrees

        # Create VAO and VBO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Set up vertex attribute pointers
        # Position (vx, vy, vz)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        # Texture coordinates (tx, ty)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        # Normals (nx, ny, nz)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

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

    def destroy(self):
        """
        Clean up the VAO and VBO.
        """
        glDeleteBuffers(1, [self.vbo])
        glDeleteVertexArrays(1, [self.vao])


if __name__ == '__main__':
    engine = GraphicsEngine("10299_Monkey-Wrench_v1_L3")