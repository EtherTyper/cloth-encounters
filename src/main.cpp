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

std::vector<std::vector<int>> facesWithVertex;


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

    facesWithVertex = std::vector<std::vector<int>>();
    facesWithVertex.resize(origQ.rows());
    for (int i = 0; i < origQ.rows(); i++) {
        facesWithVertex[i] = std::vector<int>();
    }
    for (int f = 0; f < F.rows(); f++) {
        facesWithVertex[F(f, 0)].push_back(f);
        facesWithVertex[F(f, 1)].push_back(f);
        facesWithVertex[F(f, 2)].push_back(f);
    }
    
    updateRenderGeometry();
}

void projectPinConstraints()
{
    double w = params_.pinWeight;
    for (int v : pinnedVerts) {
        Eigen::RowVector3d pinnedPos = origQ.row(v); 
        Q.row(v) = w * pinnedPos + (1.0 - w) * Q.row(v);
    }
}

void projectStretchConstraints()
{
    double w = params_.stretchWeight; 

    for (int f = 0; f < F.rows(); f++) {
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        Eigen::Vector3d x0 = Q.row(v0);
        Eigen::Vector3d x1 = Q.row(v1);
        Eigen::Vector3d x2 = Q.row(v2);

        Eigen::Vector3d r0 = origQ.row(v0);
        Eigen::Vector3d r1 = origQ.row(v1);
        Eigen::Vector3d r2 = origQ.row(v2);

        Eigen::Vector3d c  = (x0 + x1 + x2) / 3.0;
        Eigen::Vector3d c0 = (r0 + r1 + r2) / 3.0;

        Eigen::Matrix3d A, B;
        A.row(0) = x0 - c;  
        A.row(1) = x1 - c;  
        A.row(2) = x2 - c;

        B.row(0) = r0 - c0;
        B.row(1) = r1 - c0;
        B.row(2) = r2 - c0;

        Eigen::Matrix3d M = A.transpose() * B; 
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d R = U * V.transpose();

        Eigen::Vector3d new0 = c + R * (r0 - c0);
        Eigen::Vector3d new1 = c + R * (r1 - c0);
        Eigen::Vector3d new2 = c + R * (r2 - c0);

        Q.row(v0) = w * new0 + (1.0 - w) * x0;
        Q.row(v1) = w * new1 + (1.0 - w) * x1;
        Q.row(v2) = w * new2 + (1.0 - w) * x2;
    }
}

void projectBendingConstraints()
{
    double w = params_.bendingWeight;

    for (int f = 0; f < F.rows(); f++) {
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        std::set<int> adjacent_faces = std::set<int>();
        adjacent_faces.insert(facesWithVertex[F(f, 0)].cbegin(), facesWithVertex[F(f, 0)].cend());
        adjacent_faces.insert(facesWithVertex[F(f, 1)].cbegin(), facesWithVertex[F(f, 1)].cend());
        adjacent_faces.insert(facesWithVertex[F(f, 2)].cbegin(), facesWithVertex[F(f, 2)].cend());

        for (auto f2 : adjacent_faces) {
            std::set<int> vertices = { v0, v1, v2, F(f2, 0), F(f2, 1), F(f2, 2) };

            if (vertices.size() != 4) {
                continue;
            }

            auto iterator = vertices.cbegin();
            int quad_v0 = *iterator++;
            int quad_v1 = *iterator++;
            int quad_v2 = *iterator++;
            int quad_v3 = *iterator;

            Eigen::Vector3d x0 = Q.row(quad_v0);
            Eigen::Vector3d x1 = Q.row(quad_v1);
            Eigen::Vector3d x2 = Q.row(quad_v2);
            Eigen::Vector3d x3 = Q.row(quad_v3);

            Eigen::Vector3d r0 = origQ.row(quad_v0);
            Eigen::Vector3d r1 = origQ.row(quad_v1);
            Eigen::Vector3d r2 = origQ.row(quad_v2);
            Eigen::Vector3d r3 = origQ.row(quad_v3);

            Eigen::Vector3d c = (x0 + x1 + x2 + x3) / 4.0;
            Eigen::Vector3d c0 = (r0 + r1 + r2 + r3) / 4.0;

            Eigen::MatrixXd A, B;
            A.resize(4, 3);
            A.row(0) = x0 - c;
            A.row(1) = x1 - c;
            A.row(2) = x2 - c;
            A.row(3) = x3 - c;

            B.resize(4, 3);
            B.row(0) = r0 - c0;
            B.row(1) = r1 - c0;
            B.row(2) = r2 - c0;
            B.row(3) = r3 - c0;

            Eigen::Matrix3d M = A.transpose() * B;
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d U = svd.matrixU();
            Eigen::Matrix3d V = svd.matrixV();
            Eigen::Matrix3d R = U * V.transpose();

            Eigen::Vector3d new0 = c + R * (r0 - c0);
            Eigen::Vector3d new1 = c + R * (r1 - c0);
            Eigen::Vector3d new2 = c + R * (r2 - c0);
            Eigen::Vector3d new3 = c + R * (r3 - c0);

            Q.row(quad_v0) = w * new0 + (1.0 - w) * x0;
            Q.row(quad_v1) = w * new1 + (1.0 - w) * x1;
            Q.row(quad_v2) = w * new2 + (1.0 - w) * x2;
            Q.row(quad_v3) = w * new3 + (1.0 - w) * x3;
        }
    }
}

void projectPullingConstraints()
{
    if (clickedVertex < 0) return;

    double w = params_.pullingWeight;

    Eigen::RowVector3d targetPos = mousePos.transpose();
    Eigen::RowVector3d current   = Q.row(clickedVertex);

    Q.row(clickedVertex) = w * targetPos + (1.0 - w) * current;
}


void simulateOneStep()
{
    // TODO remove this when sufficiently debugged
    // running_ = false;

    Eigen::MatrixXd Qold = Q;

    Q += params_.timeStep * Qdot;

    for (int iter = 0; iter < params_.constraintIters; iter++) {

        if (params_.pinEnabled) projectPinConstraints();
        if (params_.stretchEnabled) projectStretchConstraints();
        if (params_.bendingEnabled) projectBendingConstraints();
        if (params_.pullingEnabled) projectPullingConstraints();
    }

    Qdot = (Q - Qold) / params_.timeStep;

    if (params_.gravityEnabled) {
        for (int i = 0; i < Q.rows(); i++) {
            bool pinned = false;
            if (params_.pinEnabled) {
                for (int pv : pinnedVerts) {
                    if (pv == i) { pinned = true; break; }
                }
            }
            if (!pinned) {
                Qdot(i,1) += params_.timeStep * params_.gravityG;
            }
        }
    }

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

