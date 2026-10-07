#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/readSTL.h>
#include <igl/writeSTL.h>
#include <imgui.h>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include "lattice_generator.h"

struct BoundingBox
{
    Eigen::Vector3d min;
    Eigen::Vector3d max;
    Eigen::Vector3d size;
    Eigen::Vector3d center;
};

struct FitConfiguration
{
    float head_scale = 1.0f;
    float helmet_scale = 1.0f;
    double gap = 0.005;
    Eigen::Vector3d head_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d helmet_translation = Eigen::Vector3d::Zero();
};

constexpr const char* configuration_filename = "fit_config.txt";

bool save_configuration(const FitConfiguration& config, std::string& error)
{
    std::ofstream file(configuration_filename);
    if (!file)
    {
        error = "Could not open fit_config.txt for writing.";
        return false;
    }

    file.precision(17);
    file << "ME323_FIT_CONFIG 1\n"
         << "head_scale " << config.head_scale << '\n'
         << "helmet_scale " << config.helmet_scale << '\n'
         << "gap " << config.gap << '\n'
         << "head_translation " << config.head_translation.x() << ' '
         << config.head_translation.y() << ' ' << config.head_translation.z() << '\n'
         << "helmet_translation " << config.helmet_translation.x() << ' '
         << config.helmet_translation.y() << ' ' << config.helmet_translation.z() << '\n';
    if (!file)
    {
        error = "Could not write fit_config.txt.";
        return false;
    }
    return true;
}

bool load_configuration(FitConfiguration& config, std::string& error)
{
    std::ifstream file(configuration_filename);
    if (!file)
    {
        error = "Could not open fit_config.txt.";
        return false;
    }

    std::string key;
    std::string header;
    int version = 0;
    FitConfiguration loaded;
    if (!(file >> header >> version) || header != "ME323_FIT_CONFIG" || version != 1 ||
        !(file >> key >> loaded.head_scale) || key != "head_scale" ||
        !(file >> key >> loaded.helmet_scale) || key != "helmet_scale" ||
        !(file >> key >> loaded.gap) || key != "gap" ||
        !(file >> key >> loaded.head_translation.x()
               >> loaded.head_translation.y() >> loaded.head_translation.z()) ||
        key != "head_translation" ||
        !(file >> key >> loaded.helmet_translation.x()
               >> loaded.helmet_translation.y() >> loaded.helmet_translation.z()) ||
        key != "helmet_translation")
    {
        error = "fit_config.txt is invalid or incomplete.";
        return false;
    }

    std::string extra;
    if (file >> extra)
    {
        error = "fit_config.txt contains unexpected extra data.";
        return false;
    }
    if (!std::isfinite(loaded.head_scale) || loaded.head_scale < 0.1f ||
        loaded.head_scale > 3.0f || !std::isfinite(loaded.helmet_scale) ||
        loaded.helmet_scale < 0.1f || loaded.helmet_scale > 3.0f ||
        !std::isfinite(loaded.gap) || loaded.gap < 0.0 ||
        !loaded.head_translation.allFinite() ||
        !loaded.helmet_translation.allFinite())
    {
        error = "fit_config.txt contains values outside the supported range.";
        return false;
    }

    config = loaded;
    return true;
}


BoundingBox get_bbox(const Eigen::MatrixXd& V)
{
    BoundingBox box;

    box.min = V.colwise().minCoeff();
    box.max = V.colwise().maxCoeff();

    box.size = box.max - box.min;
    box.center = (box.min + box.max) / 2.0;

    return box;
}

void fitted_vertices(
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    double gap,
    double head_scale,
    double helmet_scale,
    const Eigen::Vector3d& head_translation,
    const Eigen::Vector3d& helmet_translation_offset,
    Eigen::MatrixXd& head_vertices,
    Eigen::MatrixXd& helmet_vertices);

void update_meshes(
    igl::opengl::glfw::Viewer& viewer,
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    double gap,
    double head_scale,
    double helmet_scale,
    const Eigen::Vector3d& head_translation,
    const Eigen::Vector3d& helmet_translation_offset)
{
    Eigen::MatrixXd head_vertices;
    Eigen::MatrixXd helmet_vertices;
    fitted_vertices(
        original_head, original_helmet, original_head_box, original_helmet_box,
        automatic_helmet_scale, gap, head_scale, helmet_scale,
        head_translation, helmet_translation_offset, head_vertices, helmet_vertices);

    viewer.data_list[0].set_vertices(head_vertices);
    viewer.data_list[1].set_vertices(helmet_vertices);
    viewer.data_list[0].compute_normals();
    viewer.data_list[1].compute_normals();
}

void fitted_vertices(
    const Eigen::MatrixXd& original_head,
    const Eigen::MatrixXd& original_helmet,
    const BoundingBox& original_head_box,
    const BoundingBox& original_helmet_box,
    double automatic_helmet_scale,
    double gap,
    double head_scale,
    double helmet_scale,
    const Eigen::Vector3d& head_translation,
    const Eigen::Vector3d& helmet_translation_offset,
    Eigen::MatrixXd& head_vertices,
    Eigen::MatrixXd& helmet_vertices)
{
    head_vertices = original_head;
    head_vertices.rowwise() -= original_head_box.center.transpose();
    head_vertices *= head_scale;
    head_vertices.rowwise() +=
        (original_head_box.center + head_translation).transpose();

    const double helmet_applied_scale = automatic_helmet_scale * helmet_scale;
    const double helmet_top =
        (original_helmet_box.max.y() - original_helmet_box.center.y())
        * helmet_applied_scale;
    const double head_top =
        original_head_box.center.y()
        + (original_head_box.max.y() - original_head_box.center.y()) * head_scale
        + head_translation.y();
    Eigen::RowVector3d helmet_translation;
    helmet_translation.x() =
        original_head_box.center.x() + head_translation.x()
        + helmet_translation_offset.x();
    helmet_translation.y() =
        head_top + gap - helmet_top + helmet_translation_offset.y();
    helmet_translation.z() =
        original_head_box.center.z() + head_translation.z()
        + helmet_translation_offset.z();

    helmet_vertices = original_helmet;
    helmet_vertices.rowwise() -= original_helmet_box.center.transpose();
    helmet_vertices *= helmet_applied_scale;
    helmet_vertices.rowwise() += helmet_translation;
}


int main(int argc, char** argv)
{
    LatticeType startup_lattice = LatticeType::Gyroid;
    bool generate_only = false;
    for (int argument = 1; argument < argc; ++argument)
    {
        const std::string option = argv[argument];
        if (option == "--generate-only")
        {
            generate_only = true;
        }
        else if (option == "--lattice" && argument + 1 < argc)
        {
            const std::string requested = argv[++argument];
            if (requested == "auxetic")
            {
                startup_lattice = LatticeType::Auxetic;
            }
            else if (requested != "tpms")
            {
                std::cerr << "Unknown lattice case: " << requested
                          << ". Choose 'auxetic' or 'tpms'.\n";
                return 2;
            }
        }
        else
        {
            std::cerr << "Usage: me323 [--lattice auxetic|tpms] [--generate-only]\n";
            return 2;
        }
    }

    // Desired clearance uses the same units as the head STL (e.g. 0.005
    // for 5 mm if the model is in meters, or 5.0 if it is in millimeters).
    FitConfiguration config;

    Eigen::MatrixXd V_head, V_helmet;
    Eigen::MatrixXi F_head, F_helmet;
    Eigen::MatrixXd N_head, N_helmet;

    std::ifstream head_file(
        "data/head.stl",
        std::ios::binary
    );

    std::ifstream helmet_file(
        "data/helmet.STL",
        std::ios::binary
    );

    if (!head_file)
    {
        std::cerr << "Failed to open head.stl\n";
        return 1;
    }

    if (!helmet_file)
    {
        std::cerr << "Failed to open helmet.stl\n";
        return 1;
    }

    if (!igl::readSTL(
        head_file,
        V_head,
        F_head,
        N_head))
    {
        std::cerr << "Failed to read head.stl\n";
        return 1;
    }

    if (!igl::readSTL(
        helmet_file,
        V_helmet,
        F_helmet,
        N_helmet))
    {
        std::cerr << "Failed to read helmet.stl\n";
        return 1;
    }

    const BoundingBox head = get_bbox(V_head);
    const BoundingBox helmet = get_bbox(V_helmet);
    if (head.size.y() <= 0.0 || helmet.size.y() <= 0.0)
    {
        std::cerr << "Cannot fit meshes with zero height.\n";
        return 1;
    }
    std::string config_status = "Configuration file: fit_config.txt";
    {
        std::string error;
        if (load_configuration(config, error))
        {
            std::cout << "Loaded fit configuration from "
                      << configuration_filename << '\n';
            config_status = "Loaded fit_config.txt";
        }
        else if (error != "Could not open fit_config.txt.")
        {
            std::cerr << error << '\n';
            config_status = error;
        }
    }
    double automatic_helmet_scale =
        (head.size.y() + 2.0 * config.gap) / helmet.size.y();

    std::cout << "\n====================================\n";
    std::cout << "AUTOMATIC HELMET ALIGNMENT\n";
    std::cout << "====================================\n";

    std::cout << "Head size:\n"
              << head.size.transpose()
              << "\n";

    std::cout << "\nOriginal helmet size:\n"
              << helmet.size.transpose()
              << "\n";

    std::cout << "\nInitial automatic helmet scale:\n"
              << automatic_helmet_scale
              << "\n";

    std::cout << "\nGap:\n"
              << config.gap
              << "\n";

    // ============================================================
    // VIEWER
    // ============================================================

    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(
        V_head,
        F_head
    );
    viewer.data().show_lines = false;

    viewer.append_mesh();
    viewer.data_list[1].set_mesh(
        V_helmet,
        F_helmet
    );
    viewer.data_list[1].show_lines = false;

    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);

    std::size_t final_mesh_index = 0;
    auto show_fit_meshes = [&]()
    {
        if (final_mesh_index != 0)
        {
            viewer.data_list[final_mesh_index].set_visible(false);
        }
        viewer.data_list[0].set_visible(true);
        viewer.data_list[1].set_visible(true);
    };

    auto generate_case = [&](LatticeType type) -> bool
    {
        Eigen::MatrixXd fitted_head;
        Eigen::MatrixXd fitted_helmet;
        fitted_vertices(
            V_head, V_helmet, head, helmet, automatic_helmet_scale,
            config.gap, config.head_scale, config.helmet_scale,
            config.head_translation, config.helmet_translation,
            fitted_head, fitted_helmet);

        GeneratedMesh generated;
        std::string error;
        if (!generate_final_helmet(
                fitted_head, F_head, fitted_helmet, F_helmet,
                type, generated, error))
        {
            config_status = error;
            std::cerr << "Lattice generation failed: " << error << '\n';
            return false;
        }

        const bool auxetic = type == LatticeType::Auxetic;
        const std::string filename = auxetic
            ? "final_helmet_auxetic.stl"
            : "final_helmet_tpms.stl";
        if (!igl::writeSTL(
                filename, generated.vertices, generated.faces,
                igl::FileEncoding::Binary))
        {
            config_status = "Could not write " + filename + ".";
            std::cerr << config_status << '\n';
            return false;
        }

        if (final_mesh_index != 0)
        {
            viewer.data_list[final_mesh_index].set_visible(false);
        }
        viewer.data_list[0].set_visible(false);
        viewer.data_list[1].set_visible(false);
        viewer.append_mesh();
        final_mesh_index = viewer.data_list.size() - 1;
        viewer.data_list[final_mesh_index].set_mesh(
            generated.vertices, generated.faces);
        viewer.data_list[final_mesh_index].show_lines = false;
        viewer.core().align_camera_center(generated.vertices, generated.faces);

        config_status = "Generated " + filename + " ("
            + std::to_string(generated.vertices.rows()) + " vertices, voxel "
            + std::to_string(generated.voxel_size) + ").";
        std::cout << config_status << '\n';
        return true;
    };

    menu.callback_draw_custom_window = [&]()
    {
        ImGui::SetNextWindowSize(ImVec2(420.0f, 520.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Head / helmet fit", nullptr, ImGuiWindowFlags_NoSavedSettings);
        ImGui::Text("Helmet auto-scale: %.6g", automatic_helmet_scale);
        ImGui::TextUnformatted("Translations use the head STL's units.");

        ImGui::SliderFloat("Head scale", &config.head_scale, 0.1f, 3.0f, "%.3fx");
        const bool head_scale_finished = ImGui::IsItemDeactivatedAfterEdit();
        ImGui::SliderFloat(
            "Helmet scale", &config.helmet_scale, 0.1f, 3.0f, "%.3fx");
        const bool helmet_scale_finished = ImGui::IsItemDeactivatedAfterEdit();
        bool apply_configuration = head_scale_finished || helmet_scale_finished;

        const double translation_step = head.size.y() / 1000.0;
        const double translation_step_fast = head.size.y() / 100.0;
        ImGui::Separator();
        ImGui::TextUnformatted("Head translation (X, Y, Z)");
        ImGui::InputScalarN(
            "Head position", ImGuiDataType_Double, config.head_translation.data(), 3,
            &translation_step, &translation_step_fast, "%.6g");
        apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::Separator();
        ImGui::TextUnformatted("Helmet translation (X, Y, Z)");
        ImGui::InputScalarN(
            "Helmet position", ImGuiDataType_Double,
            config.helmet_translation.data(), 3,
            &translation_step, &translation_step_fast, "%.6g");
        apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();

        ImGui::InputDouble("Gap", &config.gap, translation_step,
                           translation_step_fast, "%.6g");
        apply_configuration |= ImGui::IsItemDeactivatedAfterEdit();

        if (ImGui::Button("Reset fit"))
        {
            config = FitConfiguration{};
            apply_configuration = true;
        }

        if (ImGui::Button("Save configuration"))
        {
            std::string error;
            if (save_configuration(config, error))
            {
                std::cout << "Saved fit configuration to "
                          << configuration_filename << '\n';
                config_status = "Saved to fit_config.txt";
            }
            else
            {
                std::cerr << error << '\n';
                config_status = error;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Load configuration"))
        {
            std::string error;
            if (load_configuration(config, error))
            {
                apply_configuration = true;
                std::cout << "Loaded fit configuration from "
                          << configuration_filename << '\n';
                config_status = "Loaded fit_config.txt";
            }
            else
            {
                std::cerr << error << '\n';
                config_status = error;
            }
        }
        ImGui::TextWrapped("%s", config_status.c_str());

        if (apply_configuration)
        {
            automatic_helmet_scale =
                (head.size.y() + 2.0 * config.gap) / helmet.size.y();
            show_fit_meshes();
            update_meshes(
                viewer, V_head, V_helmet, head, helmet,
                automatic_helmet_scale, config.gap, config.head_scale,
                config.helmet_scale, config.head_translation,
                config.helmet_translation);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Generate and export final helmet");
        if (ImGui::Button("Generate auxetic STL"))
        {
            generate_case(LatticeType::Auxetic);
        }
        ImGui::SameLine();
        if (ImGui::Button("Generate TPMS STL"))
        {
            generate_case(LatticeType::Gyroid);
        }
        if (ImGui::Button("Preview head and helmet fit"))
        {
            show_fit_meshes();
            config_status = "Showing the fitted input meshes.";
        }
        ImGui::End();
    };

    update_meshes(
        viewer, V_head, V_helmet, head, helmet,
        automatic_helmet_scale, config.gap, config.head_scale,
        config.helmet_scale, config.head_translation, config.helmet_translation);

    if (!generate_case(startup_lattice))
    {
        return 1;
    }
    if (generate_only)
    {
        return 0;
    }
    viewer.launch();

    return 0;
}