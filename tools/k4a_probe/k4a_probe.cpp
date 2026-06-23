#include <k4a/k4a.hpp>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static void save_color_ppm(const k4a::image& color, const std::string& path)
{
    int w = color.get_width_pixels();
    int h = color.get_height_pixels();
    const uint8_t* bgra = color.get_buffer();

    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";

    for (int i = 0; i < w * h; ++i)
    {
        uint8_t b = bgra[i * 4 + 0];
        uint8_t g = bgra[i * 4 + 1];
        uint8_t r = bgra[i * 4 + 2];
        out.put(static_cast<char>(r));
        out.put(static_cast<char>(g));
        out.put(static_cast<char>(b));
    }
}

static void save_depth_pgm16(const k4a::image& depth, const std::string& path)
{
    int w = depth.get_width_pixels();
    int h = depth.get_height_pixels();
    const uint16_t* z = reinterpret_cast<const uint16_t*>(depth.get_buffer());

    std::ofstream out(path, std::ios::binary);
    out << "P5\n" << w << " " << h << "\n65535\n";

    // PGM 16-bit stores big-endian values.
    for (int i = 0; i < w * h; ++i)
    {
        uint16_t v = z[i];
        out.put(static_cast<char>((v >> 8) & 0xFF));
        out.put(static_cast<char>(v & 0xFF));
    }
}

int main()
{
    try
    {
        uint32_t count = k4a::device::get_installed_count();
        std::cout << "Installed Azure Kinect devices: " << count << "\n";

        if (count == 0)
        {
            std::cerr << "No Azure Kinect devices found.\n";
            return 1;
        }

        const uint32_t device_index = 0;

        k4a::device device = k4a::device::open(device_index);
        std::cout << "Opened device index: " << device_index << "\n";
        std::cout << "Serial: " << device.get_serialnum() << "\n";

        k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        config.color_resolution = K4A_COLOR_RESOLUTION_720P;
        config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        config.camera_fps = K4A_FRAMES_PER_SECOND_30;
        config.synchronized_images_only = true;
        config.depth_delay_off_color_usec = 0;
        config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;

        k4a::calibration calibration =
            device.get_calibration(config.depth_mode, config.color_resolution);

        const auto& cp = calibration.color_camera_calibration.intrinsics.parameters.param;
        const auto& dp = calibration.depth_camera_calibration.intrinsics.parameters.param;

        std::cout << "\nColor intrinsics:\n";
        std::cout << "fx=" << cp.fx << " fy=" << cp.fy
                  << " cx=" << cp.cx << " cy=" << cp.cy << "\n";
        std::cout << "k1=" << cp.k1 << " k2=" << cp.k2 << " k3=" << cp.k3
                  << " k4=" << cp.k4 << " k5=" << cp.k5 << " k6=" << cp.k6 << "\n";
        std::cout << "p1=" << cp.p1 << " p2=" << cp.p2 << "\n";

        std::cout << "\nDepth intrinsics:\n";
        std::cout << "fx=" << dp.fx << " fy=" << dp.fy
                  << " cx=" << dp.cx << " cy=" << dp.cy << "\n";

        device.start_cameras(&config);

        k4a::capture capture;
        bool ok = device.get_capture(&capture, std::chrono::milliseconds(5000));
        if (!ok)
        {
            throw std::runtime_error("Timed out waiting for capture.");
        }

        k4a::image color = capture.get_color_image();
        k4a::image depth = capture.get_depth_image();

        if (!color || !depth)
        {
            throw std::runtime_error("Missing color or depth image.");
        }

        k4a::transformation transform(calibration);
        k4a::image depth_to_color = transform.depth_image_to_color_camera(depth);

        save_color_ppm(color, "/app/data/sdk_probe/color.ppm");
        save_depth_pgm16(depth, "/app/data/sdk_probe/depth_raw.pgm");
        save_depth_pgm16(depth_to_color, "/app/data/sdk_probe/depth_to_color.pgm");

        std::cout << "\nSaved:\n";
        std::cout << "  /app/data/sdk_probe/color.ppm\n";
        std::cout << "  /app/data/sdk_probe/depth_raw.pgm\n";
        std::cout << "  /app/data/sdk_probe/depth_to_color.pgm\n";

        device.stop_cameras();
        device.close();

        std::cout << "\nSDK probe OK.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
