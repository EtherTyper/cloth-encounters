#include "polyscope/polyscope.h"

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/pick.h"

#include <iostream>
#include <unordered_set>
#include <utility>
#include <deque>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include <igl/readOBJ.h>
#include <fstream>
#include "misc/cpp/imgui_stdlib.h"

SimParameters params_;
bool running_;

Eigen::MatrixXd origQ;
Eigen::MatrixXd Q;
Eigen::MatrixXd Qdot;
Eigen::MatrixXi F;

std::vector<int> pinnedVerts;


int clickedVertex;
double clickedDepth;
Eigen::Vector3d mousePos;

Eigen::MatrixXd renderQ;
Eigen::MatrixXi renderF;

void updateRenderGeometry()
{            
    renderQ = Q;
    renderF = F;
}


void initSimulation()
{
    if(!igl::readOBJ("meshes/rect-coarse.obj", origQ, F))
        if (!igl::readOBJ("../meshes/rect-coarse.obj", origQ, F))
        {
            std::cerr << "Couldn't read mesh file" << std::endl;
            exit(-1);
        }
    //mesh is tiny for some reason
    origQ *= 50;
    Q = origQ;  
    Qdot.resize(Q.rows(), 3);
    Qdot.setZero();

    int nverts = Q.rows();
    int topleft = -1;
    int topright = -1;
    double topleftdist = -std::numeric_limits<double>::infinity();
    double toprightdist = -std::numeric_limits<double>::infinity();
    Eigen::Vector3d tr(1, 1, 0);
    Eigen::Vector3d tl(-1, 1, 0);
    for (int i = 0; i < nverts; i++)
    {
        double disttr = tr.dot(Q.row(i));
        if (disttr > toprightdist)
        {
            toprightdist = disttr;
            topright = i;
        }
        double disttl = tl.dot(Q.row(i));
        if (disttl > topleftdist)
        {
            topleftdist = disttl;
            topleft = i;
        }
    }
    pinnedVerts.push_back(topleft);
    pinnedVerts.push_back(topright);
    

    clickedVertex = -1;
    
    updateRenderGeometry();
}


void simulateOneStep()
{
    // TODO: step time, apply gravity, enforce constraints
}

void callback()
{
    ImGui::SetNextWindowSize(ImVec2(500., 0.));
    ImGui::Begin("UI", nullptr);

    if (ImGui::Button("Recenter Camera", ImVec2(-1, 0)))
    {
        polyscope::view::resetCameraToHomeView();
    }

    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Run/Pause Sim", ImVec2(-1, 0)))
        {
            running_ = !running_;
        }
        if (ImGui::Button("Reset Sim", ImVec2(-1, 0)))
        {
            running_ = false;
            initSimulation();
        }        
    }
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Timestep", &params_.timeStep);
        ImGui::InputInt("Constraint Iters", &params_.constraintIters);
    }
    if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Gravity Enabled", &params_.gravityEnabled);
        ImGui::InputDouble("Gravity G", &params_.gravityG);
        ImGui::Checkbox("Pins Enabled", &params_.pinEnabled);
        ImGui::InputDouble("Pin Weight", &params_.pinWeight);
        ImGui::Checkbox("Stretching Enabled", &params_.stretchEnabled);
        ImGui::InputDouble("Stretching Weight", &params_.stretchWeight);
        ImGui::Checkbox("Bending Enabled", &params_.bendingEnabled);
        ImGui::InputDouble("Bending Weight", &params_.bendingWeight);
        ImGui::Checkbox("Pulling Enabled", &params_.pullingEnabled);
        ImGui::InputDouble("Pulling Weight", &params_.pullingWeight);
    }    

    ImGui::End();
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.MouseReleased[0]) {
        clickedVertex = -1;
    }
    else if (io.MouseClicked[0]) {
        glm::vec2 screenCoords{ io.MousePos.x, io.MousePos.y };
        std::pair<polyscope::Structure*, size_t> pickPair =
            polyscope::pick::evaluatePickQuery(screenCoords.x, screenCoords.y);

        if (pickPair.first != NULL)
        {
            glm::mat4 view = polyscope::view::getCameraViewMatrix();
            glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();

            if (pickPair.second < renderQ.rows())
            {
                clickedVertex = pickPair.second;
                glm::vec4 pt;
                for (int j = 0; j < 3; j++)
                    pt[j] = renderQ(clickedVertex, j);
                pt[3] = 1;
                glm::vec4 ndc = proj * view * pt;
                ndc /= ndc[3];
                clickedDepth = ndc[2];
            }
            else
            {
                int face = pickPair.second - renderQ.rows();

                int bestvert = -1;
                double bestdepth = 0;
                double bestdist = std::numeric_limits<double>::infinity();
                for (int i = 0; i < 3; i++)
                {
                    int v = renderF(face, i);
                    glm::vec4 pt;
                    for (int j = 0; j < 3; j++)
                        pt[j] = renderQ(v, j);
                    pt[3] = 1;
                    glm::vec4 ndc = proj * view * pt;
                    ndc /= ndc[3];
                    double screenx = 0.5 * (ndc[0] + 1.0);
                    double screeny = 0.5 * (1.0 - ndc[0]);
                    auto mouseXY = polyscope::view::screenCoordsToBufferInds(screenCoords);
                    double dist = (screenx - std::get<0>(mouseXY)) * (screenx - std::get<0>(mouseXY)) + (screeny - std::get<1>(mouseXY)) * (screeny - std::get<1>(mouseXY));
                    if (dist < bestdist)
                    {
                        bestdist = dist;
                        bestvert = v;
                        bestdepth = ndc[2];
                    }
                }
                clickedVertex = bestvert;
                clickedDepth = bestdepth;
            }
            mousePos = renderQ.row(clickedVertex).transpose();
        }
    }
    if (ImGui::IsMouseDragging(0))
    {
        glm::vec2 screenCoords{ io.MousePos.x, io.MousePos.y };
        int xInd, yInd;
        std::tie(xInd, yInd) = polyscope::view::screenCoordsToBufferInds(screenCoords);

        glm::mat4 view = polyscope::view::getCameraViewMatrix();
        glm::mat4 viewInv = glm::inverse(view);
        glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
        glm::mat4 projInv = glm::inverse(proj);
        
        // convert depth to world units
        glm::vec2 screenPos{ screenCoords.x / static_cast<float>(polyscope::view::windowWidth),
                            1.f - screenCoords.y / static_cast<float>(polyscope::view::windowHeight) };
        float z = clickedDepth;
        glm::vec4 clipPos = glm::vec4(screenPos * 2.0f - 1.0f, z, 1.0f);
        glm::vec4 viewPos = projInv * clipPos;
        viewPos /= viewPos.w;

        glm::vec4 worldPos = viewInv * viewPos;
        worldPos /= worldPos.w;
        for (int i = 0; i < 3; i++)
            mousePos[i] = worldPos[i];
    }
}

int main(int argc, char **argv) 
{
  polyscope::view::setWindowSize(1600, 800);
  polyscope::options::buildGui = false;
  polyscope::options::openImGuiWindowForUserCallback = false;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  polyscope::options::autocenterStructures = false;
  polyscope::options::autoscaleStructures = false;
  polyscope::options::maxFPS = -1;
  polyscope::view::setNavigateStyle(polyscope::NavigateStyle::None);
  initSimulation();

  polyscope::init();

  polyscope::state::userCallback = callback;

  while (!polyscope::render::engine->windowRequestsClose())
  {
      if (running_)
          simulateOneStep();
      updateRenderGeometry();

      auto * surf = polyscope::registerSurfaceMesh("Cloth", renderQ, renderF);
      surf->setTransparency(0.9);      

      polyscope::frameTick();
  }

  return 0;
}

