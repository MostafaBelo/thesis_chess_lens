#ifndef CHESSBOARD_DETECTOR_H
#define CHESSBOARD_DETECTOR_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <string>

namespace py = pybind11;

// Configuration struct for detection parameters
struct ChessboardDetectionConfig {
    int min_pts_needed = 15;
    int max_pts_needed = 25;
    double max_px_dist = 5.0;
    double square_eps = 3.0;
    int saddle_window_size = 4;
    int max_image_size = 500;
    double gradient_threshold_multiplier = 2.0;
    
    // Canny edge detection parameters
    double canny_low = 20;
    double canny_high = 250;
};

// Main detection function
py::array_t<float> detectChessboardCorners(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img_array,
    const ChessboardDetectionConfig& config = ChessboardDetectionConfig()
);

// Helper class for visualizing results
class ChessboardVisualizer {
public:
    static void visualizeDetection(
        const cv::Mat& img,
        const cv::Mat& corners,
        const std::vector<cv::Point>& saddle_points
    );
    
    static void visualizeGrid(
        const cv::Mat& img,
        const cv::Mat& grid_points,
        const cv::Mat& board_outline
    );
};

// Core detection pipeline functions
namespace ChessboardDetection {
    
    // Saddle point detection
    cv::Mat computeSaddle(const cv::Mat& gray_img);
    void pruneSaddle(cv::Mat& saddle, int max_points = 10000);
    cv::Mat nonMaxSuppression(const cv::Mat& img, int window = 10);
    
    // Contour processing
    void extractContours(
        const cv::Mat& edges,
        std::vector<std::vector<cv::Point>>& contours,
        std::vector<cv::Vec4i>& hierarchy
    );
    
    void filterContours(
        const std::vector<std::vector<cv::Point>>& contours_all,
        const std::vector<cv::Vec4i>& hierarchy_all,
        const cv::Mat& saddle,
        std::vector<std::vector<cv::Point>>& filtered_contours,
        std::vector<cv::Vec4i>& filtered_hierarchy,
        double area_tolerance = 0.25
    );
    
    // Grid fitting
    std::tuple<cv::Mat, cv::Mat, cv::Mat> createChessGrid(
        const cv::Mat& transform,
        int expansion = 1
    );
    
    std::tuple<cv::Mat, cv::Mat> findGridMatches(
        const cv::Mat& grid,
        const std::vector<cv::Point>& saddle_points,
        double max_distance = 5.0
    );
    
    cv::Mat fitHomography(
        const cv::Mat& ideal_grid,
        const cv::Mat& detected_grid,
        const cv::Mat& valid_mask
    );
    
    // Line detection in warped image
    std::tuple<std::vector<int>, std::vector<int>> detectBestLines(
        const cv::Mat& warped_image
    );
    
    // Geometry utilities
    bool isSquare(const std::vector<cv::Point>& contour, double eps = 3.0);
    double computeAngle(double a, double b, double c);
    std::vector<cv::Point> refineCorners(
        const std::vector<cv::Point>& contour,
        const cv::Mat& saddle,
        int window_size = 4
    );
}

#endif // CHESSBOARD_DETECTOR_H