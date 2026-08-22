// =============================================================================================================================================================================
//
// To get intrinsics of kinect 0 run:
// docker compose run --rm terminal bash -c "apt-get update && apt-get install -y g++ && g++ -O3 viki/pull_from_sdk.cpp -o extract_intrinsics -lk4a && ./extract_intrinsics 0"
//
// to instead get the ones for kinect 1 just put a 1 instead of a 0, 
// however, the ones that are already in the intrinsics_calibration.json will most likely be the same
//
// =============================================================================================================================================================================

#include <k4a/k4a.h>
#include <iostream>
#include <iomanip>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <device_index>" << std::endl;
        return 1;
    }

    uint32_t device_index = std::stoul(argv[1]);
    k4a_device_t device = NULL;

    if (k4a_device_open(device_index, &device) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to open device " << device_index << std::endl;
        return 1;
    }

    k4a_calibration_t calibration;
    // Extract for NFOV and 720p to match current server config
    if (k4a_device_get_calibration(device, K4A_DEPTH_MODE_NFOV_UNBINNED, 
                                   K4A_COLOR_RESOLUTION_720P, &calibration) != K4A_RESULT_SUCCEEDED) {
        std::cerr << "Failed to get calibration" << std::endl;
        k4a_device_close(device);
        return 1;
    }

    auto& p = calibration.color_camera_calibration.intrinsics.parameters.param;

    // Output as JSON for easy piping/parsing
    std::cout << std::fixed << std::setprecision(10);
    std::cout << "{\n";
    std::cout << "  \"fx\": " << p.fx << ",\n";
    std::cout << "  \"fy\": " << p.fy << ",\n";
    std::cout << "  \"cx\": " << p.cx << ",\n";
    std::cout << "  \"cy\": " << p.cy << ",\n";
    std::cout << "  \"dist_coeffs\": [\n";
    std::cout << "    " << p.k1 << ", " << p.k2 << ", " << p.p1 << ", " << p.p2 << ", " 
              << p.k3 << ", " << p.k4 << ", " << p.k5 << ", " << p.k6 << "\n";
    std::cout << "  ]\n";
    std::cout << "}" << std::endl;

    k4a_device_close(device);
    return 0;
}
