#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <chrono>

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include "solver.hpp"
#include "utils.hpp"

constexpr int dim = 258;
uint8_t* buf_current;
uint8_t* buf_next;

void quit(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
}

template <typename Solver>
void do_init(Solver& solver)
{
    constexpr int api = get_solver_api_version<SolverNaive<dim>>();

    if constexpr (api == 2)
        solver.init(buf_current);
}

template <typename Solver>
void do_update(Solver& solver)
{
    constexpr int api = get_solver_api_version<SolverNaive<dim>>();

    if constexpr (api == 1)
    {
        solver.update(buf_current, buf_next);
        std::swap(buf_current, buf_next);
    }
    else
    {
        solver.update();
        solver.get_results(buf_current);
    }
}

int main()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    GLFWwindow *window = glfwCreateWindow(1280, 1280, "GLFW OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);

    glfwSetKeyCallback(window, quit);

    GLuint vao_handle;
    glGenVertexArrays(1, &vao_handle);
    glBindVertexArray(vao_handle);

    glm::vec2 quad_mesh[6];
    quad_mesh[0] = glm::vec2(0.0f, 0.0f);
    quad_mesh[1] = glm::vec2(1.0f, 0.0f);
    quad_mesh[2] = glm::vec2(0.0f, 1.0f);
    quad_mesh[3] = glm::vec2(1.0f, 0.0f);
    quad_mesh[4] = glm::vec2(1.0f, 1.0f);
    quad_mesh[5] = glm::vec2(0.0f, 1.0f);
    
    GLuint vbo_handle;
    glGenBuffers(1, &vbo_handle);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_handle);
	glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(glm::vec2), quad_mesh, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0); 

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	buf_current = new uint8_t[dim * dim];
	buf_next = new uint8_t[dim * dim];

    for (int i = 0; i < dim * dim; i++)
    	buf_current[i] = 0;

    for (int i = 0; i < dim * dim; i++)
    	buf_next[i] = 0;

    // place_lidka(dim, 100, 100, buf_current);
    place_glider(dim, 10, 250, buf_current);

    GLuint tex_handle;
	glGenTextures(1, &tex_handle);
    glBindTexture(GL_TEXTURE_2D, tex_handle);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, dim, dim, 0, GL_RED, GL_UNSIGNED_BYTE, buf_current);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_handle);

	const char* vertex_source = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "out vec2 uv;\n"
    "void main()\n"
    "{\n"
    "	uv = aPos.xy;\n"
    "	gl_Position = vec4(2.0 * aPos.x - 1.0, 2.0 * aPos.y - 1.0, 0.0, 1.0);\n"
    "}\0";

    GLuint vertex_shader_handle = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex_shader_handle, 1, &vertex_source, NULL);
	glCompileShader(vertex_shader_handle);

	const char* fragment_source = "#version 330 core\n"
	"out vec4 FragColor;\n"
	"in vec2 uv;\n"
	"uniform sampler2D tex;\n"
	"void main()\n"
	"{\n"
	"	vec4 sample = texture(tex, uv);\n"
	"	float x = sample.r == 0.0 ? 0.0 : 1.0;\n"
	"	FragColor = vec4(x, x, x, 1.0f);\n"
	"}\0";

	GLuint fragment_shader_handle = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment_shader_handle, 1, &fragment_source, NULL);
	glCompileShader(fragment_shader_handle);

	GLuint shader_handle = glCreateProgram();
	glAttachShader(shader_handle, vertex_shader_handle);
	glAttachShader(shader_handle, fragment_shader_handle);
	glLinkProgram(shader_handle);

	glDeleteShader(vertex_shader_handle);
	glDeleteShader(fragment_shader_handle);  

	glUseProgram(shader_handle);

	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_handle);
	glUniform1i(glGetUniformLocation(shader_handle, "tex"), 0);

	SolverNaive<dim> solver;

    do_init(solver);

    while (!glfwWindowShouldClose(window))
    {
        do_update(solver);

#ifdef _WIN32
		Sleep(100);
#else
        sleep(0.1);
#endif

    	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, dim, dim, 0, GL_RED, GL_UNSIGNED_BYTE, buf_current);

        glfwPollEvents();

        glClearColor(0.3, 0.3, 0.3, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
    }

    glDeleteProgram(shader_handle);
    glDeleteBuffers(1, &vbo_handle);
    glDeleteVertexArrays(1, &vao_handle);
    glDeleteTextures(1, &tex_handle);

    glfwDestroyWindow(window);
    glfwTerminate();
}