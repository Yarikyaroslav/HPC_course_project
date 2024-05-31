#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <sstream>
#include <iomanip>

const double theta = 0.5;

const double G = 7.5E-05;

struct Body {
    double x, y;       // Position
    double vx, vy;     // Velocity
    
    double prev_vx, prev_vy; // Previous velocity

    double fx, fy;     // Force
    double pot_energy; // Potential energy
    double prev_fx, prev_fy; // Previous force
    double mass;       // Mass
    bool outside;
};

struct Quad {
    double x, y;       // Center
    double length;     // Length of the side

    Quad(double x, double y, double length) : x(x), y(y), length(length) {}

    bool contains(double bx, double by) const {
        return (bx >= x - length/2 && bx <= x + length/2 &&
                by >= y - length/2 && by <= y + length/2);
    }

    Quad nw() const { return Quad(x - length/4, y + length/4, length/2); }
    Quad ne() const { return Quad(x + length/4, y + length/4, length/2); }
    Quad sw() const { return Quad(x - length/4, y - length/4, length/2); }
    Quad se() const { return Quad(x + length/4, y - length/4, length/2); }
};

class Quadtree {
public:
    Quad boundary;
    Body* body;
    Quadtree* nw;
    Quadtree* ne;
    Quadtree* sw;
    Quadtree* se;
    bool divided;
    double cx, cy, cmass;

public:
    Quadtree(const Quad& boundary) : boundary(boundary), body(nullptr), nw(nullptr), ne(nullptr), sw(nullptr), se(nullptr), divided(false), cx(0), cy(0), cmass(0) {}

    ~Quadtree() {
        delete nw;
        delete ne;
        delete sw;
        delete se;
    }

    bool insert(Body* b) {
        if (!boundary.contains(b->x, b->y)) return false;

        if (divided){
            if (nw->insert(b)) return true;
            if (ne->insert(b)) return true;
            if (sw->insert(b)) return true;
            if (se->insert(b)) return true;
            return false;
        } else{
            if (!body){
                body = b;
                return true;
            } else{
                subdivide();
                auto body_copy = body;
                body = nullptr;
                nw->insert(body_copy);
                ne->insert(body_copy);
                sw->insert(body_copy);
                se->insert(body_copy);
                return insert(b);
            }
        }

        return false;
    }

    void subdivide() {
        nw = new Quadtree(boundary.nw());
        ne = new Quadtree(boundary.ne());
        sw = new Quadtree(boundary.sw());
        se = new Quadtree(boundary.se());
        divided = true;
    }

    void calculateMassDistribution() {
        if (!divided) {
            // If it's a leaf node and contains a body, set its own mass and center of mass
            if (body) {
                cmass = body->mass;
                cx = body->x;
                cy = body->y;
            }
            return;
        }

        // If it's an internal node, calculate the mass distribution from children
        nw->calculateMassDistribution();
        ne->calculateMassDistribution();
        sw->calculateMassDistribution();
        se->calculateMassDistribution();

        double total_mass = 0;
        double center_x = 0;
        double center_y = 0;

        if (nw->cmass > 0) {
            total_mass += nw->cmass;
            center_x += nw->cx * nw->cmass;
            center_y += nw->cy * nw->cmass;
        }
        if (ne->cmass > 0) {
            total_mass += ne->cmass;
            center_x += ne->cx * ne->cmass;
            center_y += ne->cy * ne->cmass;
        }
        if (sw->cmass > 0) {
            total_mass += sw->cmass;
            center_x += sw->cx * sw->cmass;
            center_y += sw->cy * sw->cmass;
        }
        if (se->cmass > 0) {
            total_mass += se->cmass;
            center_x += se->cx * se->cmass;
            center_y += se->cy * se->cmass;
        }

        cmass = total_mass;
        if (total_mass > 0) {
            cx = center_x / total_mass;
            cy = center_y / total_mass;
        }
    }

    void updateForce(Body* b) {
        if (!cmass || b == body) return;

        double dx = cx - b->x;
        double dy = cy - b->y;
        double dist = sqrt(dx * dx + dy * dy);

        if (!divided){
            double force = G*(cmass * b->mass) / (dist * dist);
            b->pot_energy -= G*(cmass * b->mass) / dist;
            b->fx += force * dx / dist;
            b->fy += force * dy / dist;
            return;
        }
        
        if ((boundary.length / dist) < theta) {
            double force = G*(cmass * b->mass) / (dist * dist);
            b->pot_energy -= G*(cmass * b->mass) / dist;
            b->fx += force * dx / dist;
            b->fy += force * dy / dist;
        } else {
            nw->updateForce(b);
            ne->updateForce(b);
            sw->updateForce(b);
            se->updateForce(b);
        }
    }
};


void calculateForces(std::vector<Body>& bodies, Quadtree& tree) {
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); ++i) {
        bodies[i].fx = 0;
        bodies[i].fy = 0;
        bodies[i].pot_energy = 0;
        if (!bodies[i].outside){
            tree.updateForce(&bodies[i]);
        }
    }
}

void updateBodies(std::vector<Body>& bodies, double dt) {
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); ++i) {
        if (!bodies[i].outside){
            bodies[i].x += 3/2 * bodies[i].vx * dt - 1/2 * bodies[i].prev_vx * dt;
            bodies[i].y += 3/2 * bodies[i].vy * dt - 1/2 * bodies[i].prev_vy * dt;
            double new_vx = bodies[i].vx + 3/2 * bodies[i].fx / bodies[i].mass * dt - 1/2 * bodies[i].prev_fx / bodies[i].mass * dt;
            double new_vy = bodies[i].vy + 3/2 * bodies[i].fy / bodies[i].mass * dt - 1/2 * bodies[i].prev_fy / bodies[i].mass * dt;

            bodies[i].prev_vx = bodies[i].vx;
            bodies[i].prev_vy = bodies[i].vy;
            bodies[i].prev_fx = bodies[i].fx;
            bodies[i].prev_fy = bodies[i].fy;
            bodies[i].vx = new_vx;
            bodies[i].vy = new_vy;
        }
    }
}

void updateBodiesEuler(std::vector<Body>& bodies, double dt) {
    #pragma omp parallel for
    for (int i = 0; i < bodies.size(); ++i) {
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        double new_vx = bodies[i].vx + bodies[i].fx / bodies[i].mass * dt;
        double new_vy = bodies[i].vy + bodies[i].fy / bodies[i].mass * dt;

        bodies[i].prev_vx = bodies[i].vx;
        bodies[i].prev_vy = bodies[i].vy;

        bodies[i].prev_fx = bodies[i].fx;
        bodies[i].prev_fy = bodies[i].fy;
        bodies[i].vx = new_vx;
        bodies[i].vy = new_vy;
        
    }
}

double computeKineticEnergy(const std::vector<Body>& bodies) {
    double kineticEnergy = 0.0;

    // Compute kinetic energy
    #pragma omp parallel for reduction(+:kineticEnergy)
    for (int i = 0; i < bodies.size(); ++i) {
        kineticEnergy += 0.5 * bodies[i].mass * (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy);
    }

    return kineticEnergy;
}

double computePotentialEnergy(const std::vector<Body>& bodies) {
    double potentialEnergy = 0.0;

    // Compute potential energy 
    #pragma omp parallel for reduction(+:potentialEnergy)
    for (int i = 0; i < bodies.size(); ++i) {
        potentialEnergy += bodies[i].pot_energy;
    }

    return potentialEnergy;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double totalEnergy = 0.0;

    // Compute kinetic energy
    #pragma omp parallel for reduction(+:totalEnergy)
    for (int i = 0; i < bodies.size(); ++i) {
        double kineticEnergy = 0.5 * bodies[i].mass * (bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy);
        totalEnergy += kineticEnergy + bodies[i].pot_energy;
    }

    return totalEnergy;
}

std::vector<Body> loadInitialConditions(const std::string& filename) {
    std::vector<Body> bodies;
    std::ifstream file(filename);
    std::string line;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        Body body;
        if (iss >> body.x >> body.y >> body.vx >> body.vy >> body.mass) {
            body.fx = 0;
            body.fy = 0;
            body.prev_fx = 0;
            body.prev_fy = 0;
            body.prev_vx = 0;
            body.prev_vy = 0;
            body.outside = false;
            bodies.push_back(body);
        }
    }
    return bodies;
}


void saveFinalPositions(const std::string& filename, const std::vector<Body>& bodies) {
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(6); 
    for (const Body& body : bodies) {
        file << body.x << " " << body.y << "\n";
    }
}

int main() {
    const double dt = 1.0E-05;

    const int Nsteps = 500000;
    // Load initial conditions from file
    std::vector<Body> bodies = loadInitialConditions("initial_conditions.txt");

    // Simulation loop
    for (int step = 0; step < Nsteps+1; ++step) {
        // Create the quadtree
        Quad boundary(0, 0, 15);
        Quadtree tree(boundary);
        
        for (Body& body : bodies) {
            if(!tree.insert(&body)){
                body.outside = true;
            }
        }

        // Calculate mass distribution in the quadtree
        tree.calculateMassDistribution();

        // Calculate forces in parallel
        calculateForces(bodies, tree);


        // if (step%100==0){
        // double totalEnergy = computeTotalEnergy(bodies);
        // std::cout << step << " " << totalEnergy << std::endl;
        // }

        // Update body positions
        if (step==0){
            updateBodiesEuler(bodies, dt);
        } else{
            updateBodies(bodies, dt);
        }

        if (step%10000==0){
        //    double KineticEnergy = computeKineticEnergy(bodies);
        //    double PotentialEnergy = computePotentialEnergy(bodies);
        //    std::cout << step << " " << KineticEnergy << " " << PotentialEnergy << std::endl;
            saveFinalPositions("output_" + std::to_string(step/10) + ".txt", bodies);
            std::cout << step << std::endl;
        }

    }

    

    return 0;
}