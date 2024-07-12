#include <Python.h>
#include <SFML/Graphics.hpp>
#include <vector>
#include <cstdlib> // For rand()
#include <iostream>
#include <cmath>
#include "particle.hpp"
#include "forces.hpp"
#include <array> // Include for std::array

// Initialize the Python interpreter
void initializePython() {
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./PINN')"); // Adjust the path as needed

    PyRun_SimpleString("from inference import pinn_inference");
    PyRun_SimpleString("pinn_inference(0.0)"); // Warm-up call with dummy data
}

// Finalize the Python interpreter
void finalizePython() {
    Py_Finalize();
}

// Function to call the PINN model in Python
float callPINNModel(float input) {
    PyObject* pName = PyUnicode_DecodeFSDefault("inference");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != nullptr) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "pinn_inference");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject* pArgs = PyTuple_Pack(1, PyFloat_FromDouble(input));
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != nullptr) {
                float result = static_cast<float>(PyFloat_AsDouble(pValue));
                Py_DECREF(pValue);
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                return result;
            } else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                throw std::runtime_error("Call to Python function failed.");
            }
        } else {
            if (PyErr_Occurred()) PyErr_Print();
            Py_XDECREF(pFunc);
            Py_DECREF(pModule);
            throw std::runtime_error("Cannot find function pinn_inference.");
        }
    } else {
        PyErr_Print();
        throw std::runtime_error("Failed to load Python module.");
    }
}

const int cellWidth = 100;
const int cellHeight = 100;
std::vector<std::vector<std::vector<Particle*>>> grid;
std::vector<Particle> particles;

void RenderWall(sf::RenderWindow& window, const sf::RectangleShape& wall) {
    window.draw(wall);
}

bool isColliding(Particle& particle1, Particle& particle2) {
    float dist = sqrt(pow((particle1[0] - particle2[0]), 2) + pow((particle1[1] - particle2[1]), 2));
    return dist < particle1.getRadius() + particle2.getRadius();
}

void checkAndBounceOffWalls(Particle& particle, float windowWidth, float windowHeight) {
    float radius = particle.getRadius();  // Particle radius
    float restitution = 0.9f; // Coefficient of restitution

    if (particle[0] - radius < 0) {
        particle.setVelocityX(-particle.getVelocityX() * restitution);
        particle[0] = radius;
    } else if (particle[0] + radius > windowWidth) {
        particle.setVelocityX(-particle.getVelocityX() * restitution);
        particle[0] = windowWidth - radius;
    }

    if (particle[1] - radius < 0) {
        particle.setVelocityY(callPINNModel(particle.getVelocityY()) * restitution);
        particle[1] = radius;
    } else if (particle[1] + radius > windowHeight) {
        particle.setVelocityY(callPINNModel(particle.getVelocityY()) * restitution);
        particle[1] = windowHeight - radius;
    }
}

void resolveCollision(Particle& particle1, Particle& particle2) {
    float restitution = 0.9f;
    float dvx = particle1.getVelocityX() - particle2.getVelocityX();
    float dvy = particle1.getVelocityY() - particle2.getVelocityY();
    float dx = particle1[0] - particle2[0];
    float dy = particle1[1] - particle2[1];

    float overlap = particle1.getRadius() + particle2.getRadius() - sqrt(pow(dx, 2) + pow(dy, 2));
    if (overlap > 0) {
        float overlapX = overlap * dx / (2 * sqrt(pow(dx, 2) + pow(dy, 2)));
        float overlapY = overlap * dy / (2 * sqrt(pow(dy, 2) + pow(dx, 2)));

        particle1[0] += overlapX;
        particle1[1] += overlapY;
        particle2[0] -= overlapX;
        particle2[1] -= overlapY;
    }

    float factor = 2.0 / (particle1.getMass() + particle2.getMass());
    float dotProduct = (dvx * dx + dvy * dy) / (pow(dx, 2) + pow(dy, 2));
    dotProduct *= restitution;

    particle1.setVelocityX(particle1.getVelocityX() - factor * particle2.getMass() * dotProduct * dx);
    particle1.setVelocityY(particle1.getVelocityY() - factor * particle2.getMass() * dotProduct * dy);
    particle2.setVelocityX(particle2.getVelocityX() + factor * particle1.getMass() * dotProduct * dx);
    particle2.setVelocityY(particle2.getVelocityY() + factor * particle1.getMass() * dotProduct * dy);
}

bool isOverlapping(const Particle& newParticle, const std::vector<Particle>& particles) {
    for (const auto& particle : particles) {
        float dist = sqrt(pow(newParticle[0] - particle[0], 2) + pow(newParticle[1] - particle[1], 2));
        if (dist < particle.getRadius() + newParticle.getRadius()) {
            return true;
        }
    }
    return false;
}

void updateGrid() {
    for (auto &column : grid) {
        for (auto &cell : column) {
            cell.clear();
        }
    }

    for (Particle& particle : particles) {
        int gridX = particle.getX() / cellWidth;
        int gridY = particle.getY() / cellHeight;
        if(gridX >= 0 && gridX < grid.size() && gridY >= 0 && gridY < grid[gridX].size()) {
            grid[gridX][gridY].push_back(&particle);
        }
    }
}

int main() {
    initializePython();

    srand(static_cast<unsigned int>(time(0)));
    auto window = sf::RenderWindow{{1920u, 1080u}, "CMake SFML Project"};
    window.setFramerateLimit(144);

    int gridRows = 1080 / cellHeight;
    int gridCols = 1920 / cellWidth;
    grid.resize(gridCols, std::vector<std::vector<Particle*>>(gridRows));
    int numParticles = 100;

    for (int i = 0; i < numParticles; ++i) {
        bool positionFound = false;
        Particle newParticle;

        while (!positionFound) {
            float x = static_cast<float>(rand() % 1920);
            float y = static_cast<float>(rand() % 1080);
            float vx = static_cast<float>(rand() % 100) / 10.0f - 50.0f;
            float vy = static_cast<float>(rand() % 100) / 10.0f - 50.0f;
            newParticle = Particle(x, y, vx, vy, 1.0f, 3.0f);

            if (!isOverlapping(newParticle, particles)) {
                positionFound = true;
            }
        }

        particles.emplace_back(newParticle);
    }

    sf::RectangleShape wall(sf::Vector2f(200, 200));
    wall.setPosition(1920, 1080);
    wall.setFillColor(sf::Color::Red);

    float dt = 1.0f / 60.0f;

    while (window.isOpen()) {
        for (auto event = sf::Event{}; window.pollEvent(event);) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        for (Particle& particle : particles) {
            Force::ApplyGravity(particle, 9.8f, dt);
            checkAndBounceOffWalls(particle, 800.0f, 800.0f);
            particle.Update(dt);
        }

        updateGrid();

        for (size_t x = 0; x < grid.size(); ++x) {
            for (size_t y = 0; y < grid[x].size(); ++y) {
                for (Particle* particle : grid[x][y]) {
                    for (Particle* other : grid[x][y]) {
                        if (particle != other && isColliding(*particle, *other)) {
                            resolveCollision(*particle, *other);
                        }
                    }

                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            int nx = x + dx;
                            int ny = y + dy;
                            if (nx >= 0 && ny >= 0 && nx < grid.size() && ny < grid[nx].size()) {
                                for (Particle* neighbor : grid[nx][ny]) {
                                    if (particle != neighbor && isColliding(*particle, *neighbor)) {
                                        resolveCollision(*particle, *neighbor);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        window.clear();
        for (const Particle& particle : particles) {
            particle.Render(window);
        }

        RenderWall(window, wall);

        window.display();
    }

    finalizePython();
}
