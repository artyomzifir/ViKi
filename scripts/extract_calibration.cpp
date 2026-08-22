#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <k4a/k4a.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <device_index>" << std::endl;
        return 1;
    }

    uint32_t device_index = std::stoul(argv[1]);
    k4a_device_t device = nullptr;
    if (k4a_device_open(device_index, &device) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to open device " << device_index << std::endl;
        return 1;
    }

    k4a_calibration_t calibration;
    if (k4a_device_get_calibration(device, K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_COLOR_RESOLUTION_720P, &calibration) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to get calibration" << std::endl;
        k4a_device_close(device);
        return 1;
    }

    std::cout << "{" << std::endl;
    std::cout << "  \"device_index\": " << device_index << "," << std::endl;

    // 1. Extract Intrinsics for Color and Depth
    auto extract_intrinsics = [&](k4a_calibration_type_t cam_type, const std::string& name) {
        k4a_float3_t p0 = {0.0f, 0.0f, 1000.0f};
        k4a_float2_t pix0;
        int valid0;
        k4a_calibration_3d_to_2d(&calibration, &p0, cam_type, cam_type, &pix0, &valid0);

        k4a_float3_t pX = {1000.0f, 0.0f, 1000.0f};
        k4a_float2_t pixX;
        int validX;
        k4a_calibration_3d_to_2d(&calibration, &pX, cam_type, cam_type, &pixX, &validX);

        k4a_float3_t pY = {0.0f, 1000.0f, 1000.0f};
        k4a_float2_t pixY;
        int validY;
        k4a_calibration_3d_to_2d(&calibration, &pY, cam_type, cam_type, &pixY, &validY);

        float cx = ((float*) &pix0)[0];
        float cy = ((float*) &pix0)[1];
        float fx = ((float*) &pixX)[0] - cx;
        float fy = ((float*) &pixY)[1] - cy;

        std::cout << "  \"" << name << "_intrinsics\": {" << std::endl;
        std::cout << "    \"fx\": " << fx << "," << std::endl;
        std::cout << "    \"fy\": " << fy << "," << std::endl;
        std::cout << "    \"cx\": " << cx << "," << std::endl;
        std::cout << "    \"cy\": " << cy << std::endl;
        std::cout << "  }," << std::endl;
    };

    extract_intrinsics(K4A_CALIBRATION_TYPE_COLOR, "color");
    extract_intrinsics(K4A_CALIBRATION_TYPE_DEPTH, "depth");

    // 2. Extract Extrinsics (Depth -> Color)
    k4a_float3_t origin = {0.0f, 0.0f, 0.0f};
    k4a_float3_t t_origin;
    k4a_calibration_3d_to_3d(&calibration, &origin, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &t_origin);

    k4a_float3_t bx = {1.0f, 0.0f, 0.0f};
    k4a_float3_t tx;
    k4a_calibration_3d_to_3d(&calibration, &bx, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &tx);

    k4a_float3_t by = {0.0f, 1.0f, 0.0f};
    k4a_float3_t ty;
    k4a_calibration_3d_to_3d(&calibration, &by, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &ty);

    k4a_float3_t bz = {0.0f, 0.0f, 1.0f};
    k4a_float3_t tz;
    k4a_calibration_3d_to_3d(&calibration, &bz, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_COLOR, &tz);

    std::cout << "  \"extrinsics\": {" << std::endl;
    std::cout << "    \"rotation\": [";
    for (int i = 0; i < 3; ++i) {
        std::cout << "[";
        std::cout << ((float*)&tx)[i] - ((float*)&t_origin)[i] << ", " 
                  << ((float*)&ty)[i] - ((float*)&t_origin)[i] << ", " 
                  << ((float*)&tz)[i] - ((float*)&t_origin)[i] << "]";
        if (i < 2) std::cout << ", ";
    }
    std::cout << "],";
    
    std::cout << "\n    \"translation\": [";
    std::cout << ((float*)&t_origin)[0] << ", " << ((float*)&t_origin)[1] << ", " << ((float*)&t_origin)[2] << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << "}" << std::endl;

    k4a_device_close(device);
    return 0;
}
