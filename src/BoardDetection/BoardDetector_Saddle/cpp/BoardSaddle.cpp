#include "BoardSaddle.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <string>
#include <tuple>

using namespace cv;
using namespace std;

namespace {
    // RAII wrapper for numpy array to ensure proper lifetime
    template <typename T>
    class NumpyArrayWrapper {
    public:
        NumpyArrayWrapper(py::array_t<T, py::array::c_style | py::array::forcecast> arr) 
            : array_(arr), info_(arr.request()) {
            if (info_.ndim != 3) {
                throw std::runtime_error("Expected 3D array, got " + std::to_string(info_.ndim) + "D");
            }
        }
        
        Mat toMat() {
            // Create Mat with proper type based on channels
            int rows = info_.shape[0];
            int cols = info_.shape[1];
            int channels = info_.shape[2];
            
            int cv_type;
            if (channels == 1) cv_type = CV_8UC1;
            else if (channels == 3) cv_type = CV_8UC3;
            else if (channels == 4) cv_type = CV_8UC4;
            else throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
            
            Mat mat(rows, cols, cv_type, info_.ptr);
            return mat.clone(); // Clone to avoid lifetime issues
        }
        
    private:
        py::array_t<T> array_;  // Keep array alive
        py::buffer_info info_;
    };
    
    // Utility: Safe matrix access with bounds checking
    template<typename T>
    T safeMatAccess(const Mat& mat, int row, int col, T default_val = T()) {
        if (row >= 0 && row < mat.rows && col >= 0 && col < mat.cols) {
            return mat.at<T>(row, col);
        }
        return default_val;
    }
}

// Visualization function
void visualizeChessGrids(
    const Mat& img,
    const Mat& ideal_grid,
    const Mat& warped_grid,
    const Mat& new_grid,
    const Mat& grid_good,
    const vector<Point>& spts
) {
    Mat vis;
    if (!img.empty()) {
        vis = img.clone();
    }
    else {
        vis = Mat::zeros(600, 600, CV_8UC3);
    }

    for (int i = 0; i < ideal_grid.rows; i++) {
        circle(vis, Point2f(ideal_grid.at<float>(i, 0), ideal_grid.at<float>(i, 1)), 4, Scalar(255, 0, 0), -1);
    }

    for (int i = 0; i < warped_grid.rows; i++) {
        circle(vis, Point2f(warped_grid.at<float>(i, 0), warped_grid.at<float>(i, 1)), 4, Scalar(0, 255, 0), -1);
    }

    for (int i = 0; i < new_grid.rows; i++) {
        bool is_good = grid_good.at<uchar>(i, 0) > 0;
        if (is_good) {
            circle(vis, Point2f(new_grid.at<float>(i, 0), new_grid.at<float>(i, 1)), 4, Scalar(0, 0, 255), -1);
        }
    }

    for (const auto& pt : spts) {
        circle(vis, pt, 3, Scalar(0, 255, 255), -1);
    }

    imshow("Chess Grid Visualization", vis);
    waitKey(0);
}

vector<Point> updateCorners(const vector<Point>& contour, const Mat& saddle) {
    int ws = 4;
    vector<Point> new_contour = contour;

    for (size_t i = 0; i < contour.size(); i++) {
        int cc = contour[i].x;
        int rr = contour[i].y;

        int rl = max(0, rr - ws);
        int cl = max(0, cc - ws);

        Mat window = saddle(Range(rl, min(saddle.rows, rr + ws + 1)),
            Range(cl, min(saddle.cols, cc + ws + 1)));

        double minVal, maxVal;
        Point maxLoc;
        minMaxLoc(window, &minVal, &maxVal, nullptr, &maxLoc);
        int br = maxLoc.y - min(ws, rl);
        int bc = maxLoc.x - min(ws, cl);

        if (maxVal > 0)
            new_contour[i] = Point(cc + bc, rr + br);
        else
            return {};
    }
    return new_contour;
}

double getAngle(double a, double b, double c) {
    double k = (a * a + b * b - c * c) / (2 * a * b);
    if (k < -1) k = -1;
    if (k > 1) k = 1;
    return acos(k) * 180.0 / CV_PI;
}

bool is_square(const vector<Point>& cnt, double eps = 3.0) {
    if (cnt.size() != 4) return false;

    auto dist = [](const Point& p1, const Point& p2) {
        return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    };

    double dd0 = dist(cnt[0], cnt[1]);
    double dd1 = dist(cnt[1], cnt[2]);
    double dd2 = dist(cnt[2], cnt[3]);
    double dd3 = dist(cnt[3], cnt[0]);

    double xa = dist(cnt[0], cnt[2]);
    double xb = dist(cnt[1], cnt[3]);

    double ta = getAngle(dd3, dd0, xb);
    double tb = getAngle(dd0, dd1, xa);
    double tc = getAngle(dd1, dd2, xb);
    double td = getAngle(dd2, dd3, xa);
    double angle_sum = round(ta + tb + tc + td);

    bool good_angles = (ta > 40 && ta < 140) && (tb > 40 && tb < 140) &&
        (tc > 40 && tc < 140) && (td > 40 && td < 140);

    double dda = max(dd0 / dd1, dd1 / dd0);
    double ddb = max(dd1 / dd2, dd2 / dd1);
    double ddc = max(dd2 / dd3, dd3 / dd2);
    double ddd = max(dd3 / dd0, dd0 / dd3);

    bool good_side_ratios = (dda < eps && ddb < eps && ddc < eps && ddd < eps);

    return good_angles && good_side_ratios && abs(angle_sum - 360) < 5;
}

Mat getSaddle(const Mat& gray_img) {
    Mat img;
    gray_img.convertTo(img, CV_64F);

    Mat gx, gy, gxx, gyy, gxy;
    Sobel(img, gx, CV_64F, 1, 0);
    Sobel(img, gy, CV_64F, 0, 1);
    Sobel(gx, gxx, CV_64F, 1, 0);
    Sobel(gy, gyy, CV_64F, 0, 1);
    Sobel(gx, gxy, CV_64F, 0, 1);

    Mat S = gxx.mul(gyy) - gxy.mul(gxy);
    return S;
}

void pruneSaddle(Mat& s) {
    double thresh = 128;
    int score = countNonZero(s > 0);
    while (score > 10000) {
        thresh *= 2;
        s.setTo(0, s < thresh);
        score = countNonZero(s > 0);
    }
}

Mat nonmax_sup(const Mat& img, int win = 10) {
    Mat img_sup = Mat::zeros(img.size(), CV_64F);

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.at<double>(y, x) != 0) {
                int y0 = max(0, y - win);
                int y1 = min(img.rows, y + win + 1);
                int x0 = max(0, x - win);
                int x1 = min(img.cols, x + win + 1);
                Mat cell = img(Range(y0, y1), Range(x0, x1));
                double val = img.at<double>(y, x);
                double maxVal;
                minMaxLoc(cell, nullptr, &maxVal);
                if (maxVal == val)
                    img_sup.at<double>(y, x) = val;
            }
        }
    }
    return img_sup;
}

vector<vector<Point>> simplifyContours(const vector<vector<Point>>& contours) {
    vector<vector<Point>> simplified;
    for (const auto& cnt : contours) {
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * arcLength(cnt, true), true);
        simplified.push_back(approx);
    }
    return simplified;
}

void getContours(const Mat& img, const Mat& edges, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat edges_gradient;
    morphologyEx(edges, edges_gradient, MORPH_GRADIENT, kernel);

    vector<vector<Point>> tmp_contours;
    vector<Vec4i> tmp_hierarchy;
    findContours(edges_gradient.clone(), tmp_contours, tmp_hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    contours = simplifyContours(tmp_contours);
    hierarchy = tmp_hierarchy;
}

void pruneContours(const vector<vector<Point>>& contours_all, const vector<Vec4i>& hierarchy_all,
    const Mat& saddle, vector<vector<Point>>& new_contours, vector<Vec4i>& new_hierarchy) {

    for (size_t i = 0; i < contours_all.size(); i++) {
        auto cnt = contours_all[i];
        Vec4i h = hierarchy_all[i];

        if (h[2] != -1) continue;
        if (cnt.size() != 4) continue;
        if (contourArea(cnt) < 64) continue;

        vector<Point> updated_cnt = updateCorners(cnt, saddle);
        if (updated_cnt.empty()) continue;

        new_contours.push_back(updated_cnt);
        new_hierarchy.push_back(h);
    }

    if (new_contours.empty()) return;

    vector<double> areas;
    for (const auto& cnt : new_contours)
        areas.push_back(contourArea(cnt));

    double median_area;
    sort(areas.begin(), areas.end());
    median_area = areas[areas.size() / 2];

    vector<vector<Point>> filtered_contours;
    vector<Vec4i> filtered_hierarchy;
    for (size_t i = 0; i < new_contours.size(); i++) {
        if (areas[i] >= 0.25 * median_area && areas[i] <= 2.0 * median_area) {
            filtered_contours.push_back(new_contours[i]);
            filtered_hierarchy.push_back(new_hierarchy[i]);
        }
    }

    new_contours = filtered_contours;
    new_hierarchy = filtered_hierarchy;
}

Mat getIdentityGrid(int N) {
    Mat grid(N * N, 2, CV_32F);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            grid.at<float>(y * N + x, 0) = float(x);
            grid.at<float>(y * N + x, 1) = float(y);
        }
    }
    return grid;
}

tuple<Mat, Mat, Mat> makeChessGrid(const Mat& M, int N = 1) {
    Mat ideal_grid = getIdentityGrid(2 + 2 * N);
    ideal_grid.convertTo(ideal_grid, CV_32F);
    ideal_grid -= Scalar(N, N);

    Mat ones_col = Mat::ones(ideal_grid.rows, 1, CV_32F);
    Mat ideal_grid_pad;
    hconcat(ideal_grid, ones_col, ideal_grid_pad);

    Mat M32;
    M.convertTo(M32, CV_32F);

    Mat grid_h = M32 * ideal_grid_pad.t();
    Mat grid_t = grid_h.t();

    Mat grid(grid_t.rows, 2, CV_32F);
    for (int i = 0; i < grid_t.rows; i++) {
        float xh = grid_t.at<float>(i, 0);
        float yh = grid_t.at<float>(i, 1);
        float wh = grid_t.at<float>(i, 2);

        grid.at<float>(i, 0) = xh / wh;
        grid.at<float>(i, 1) = yh / wh;
    }

    return make_tuple(grid, ideal_grid, M.clone());
}

tuple<Mat, Mat, Mat> getInitChessGrid(const vector<Point>& quad) {
    Mat quadMat(4, 2, CV_32F);
    for (int i = 0; i < 4; i++) {
        quadMat.at<float>(i, 0) = quad[i].x;
        quadMat.at<float>(i, 1) = quad[i].y;
    }
    Mat quadA = (Mat_<float>(4, 2) << 0, 1, 1, 1, 1, 0, 0, 0);
    Mat M = getPerspectiveTransform(quadA, quadMat);
    return makeChessGrid(M, 1);
}

tuple<Mat, Mat> findGoodPoints(const Mat& grid, const vector<Point>& spts, double max_px_dist = 5.0) {
    Mat new_grid = grid.clone();
    Mat grid_good = Mat::zeros(grid.rows, 1, CV_8U);
    set<string> chosen_spts;

    auto hash_pt = [](int x, int y) { return to_string(x) + "_" + to_string(y); };

    for (int i = 0; i < grid.rows; i++) {
        Point2f pt(grid.at<float>(i, 0), grid.at<float>(i, 1));
        Point2f best_pt = pt;
        double best_dist = 1e9;

        for (const auto& sp : spts) {
            double dx = sp.x - pt.x;
            double dy = sp.y - pt.y;
            double dist = dx * dx + dy * dy;
            if (dist < best_dist) {
                best_dist = dist;
                best_pt = Point2f(sp.x, sp.y);
            }
        }
        best_dist = sqrt(best_dist);

        string h = hash_pt(int(best_pt.x), int(best_pt.y));
        if (chosen_spts.count(h)) {
            best_dist = max_px_dist;
        } else {
            chosen_spts.insert(h);
        }

        if (best_dist < max_px_dist) {
            new_grid.at<float>(i, 0) = best_pt.x;
            new_grid.at<float>(i, 1) = best_pt.y;
            grid_good.at<uchar>(i, 0) = 1;
        }
    }
    return make_tuple(new_grid, grid_good);
}

Mat generateNewBestFit(const Mat& grid_ideal, const Mat& grid, const Mat& grid_good) {
    vector<Point2f> a, b;
    for (int i = 0; i < grid.rows; i++) {
        if (grid_good.at<uchar>(i, 0)) {
            a.push_back(Point2f(grid_ideal.at<float>(i, 0), grid_ideal.at<float>(i, 1)));
            b.push_back(Point2f(grid.at<float>(i, 0), grid.at<float>(i, 1)));
        }
    }
    if (a.size() < 4) return Mat();
    Mat M = findHomography(a, b, RANSAC);
    return M;
}

tuple<Mat, Mat, Mat, Mat, Mat, Mat, Mat> getGrads(const Mat& img) {
    Mat img_blur;
    blur(img, img_blur, Size(5, 5));
    Mat gx, gy;
    Sobel(img_blur, gx, CV_64F, 1, 0);
    Sobel(img_blur, gy, CV_64F, 0, 1);

    Mat grad_mag = gx.mul(gx) + gy.mul(gy);
    Mat grad_phase;
    phase(gx, gy, grad_phase);

    double gradient_mask_threshold = 2 * mean(grad_mag)[0];
    Mat grad_phase_masked = grad_phase.clone();
    for (int y = 0; y < grad_mag.rows; y++) {
        for (int x = 0; x < grad_mag.cols; x++) {
            if (grad_mag.at<double>(y, x) < gradient_mask_threshold)
                grad_phase_masked.at<double>(y, x) = NAN;
        }
    }
    return make_tuple(grad_mag, grad_phase_masked, grad_phase, gx, gy, gx, gy);
}

tuple<vector<int>, vector<int>> getBestLines(const Mat& img_warped) {
    Mat grad_mag, grad_phase_masked, grad_phase, gx, gy, _, __;
    tie(grad_mag, grad_phase_masked, grad_phase, gx, gy, _, __) = getGrads(img_warped);

    Mat gx_pos = gx.clone(); 
    gx_pos.setTo(0, gx_pos < 0);
    Mat gx_neg = -gx; 
    gx_neg.setTo(0, gx_neg < 0);
    Mat gy_pos = gy.clone(); 
    gy_pos.setTo(0, gy_pos < 0);
    Mat gy_neg = -gy; 
    gy_neg.setTo(0, gy_neg < 0);

    Mat score_x = Mat::zeros(1, gx.cols, CV_64F);
    Mat score_y = Mat::zeros(gy.rows, 1, CV_64F);

    for (int i = 0; i < gx.cols; i++)
        score_x.at<double>(0, i) = sum(gx_pos.col(i))[0] * sum(gx_neg.col(i))[0];
    for (int i = 0; i < gy.rows; i++)
        score_y.at<double>(i, 0) = sum(gy_pos.row(i))[0] * sum(gy_neg.row(i))[0];

    vector<vector<int>> a;
    for (int offset = 1; offset <= 8; offset++) {
        vector<int> pts(7);
        for (int j = 0; j < 7; j++) 
            pts[j] = (offset + j + 1) * 32;
        a.push_back(pts);
    }

    vector<double> scores_x(a.size(), 0), scores_y(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++) {
        for (auto idx : a[i]) {
            if (idx < score_x.cols)
                scores_x[i] += score_x.at<double>(0, idx);
        }
        for (auto idx : a[i]) {
            if (idx < score_y.rows)
                scores_y[i] += score_y.at<double>(idx, 0);
        }
    }

    auto best_x = max_element(scores_x.begin(), scores_x.end()) - scores_x.begin();
    auto best_y = max_element(scores_y.begin(), scores_y.end()) - scores_y.begin();
    return make_tuple(a[best_x], a[best_y]);
}

Mat loadImage(const string& filepath) {
    Mat img_orig = imread(filepath, IMREAD_GRAYSCALE);
    if (img_orig.empty()) {
        cerr << "Failed to load image: " << filepath << endl;
        return Mat();
    }

    int img_width = img_orig.cols;
    int img_height = img_orig.rows;

    double aspect_ratio = min(500.0 / img_width, 500.0 / img_height);
    int new_width = static_cast<int>(img_width * aspect_ratio);
    int new_height = static_cast<int>(img_height * aspect_ratio);

    Mat img_resized;
    resize(img_orig, img_resized, Size(new_width, new_height), 0, 0, INTER_LINEAR);

    return img_resized;
}

tuple<Mat, Mat, Mat, Mat, vector<Point>> findChessboard(const Mat& img, int min_pts_needed = 15, int max_pts_needed = 25) {
    Mat blur_img;
    blur(img, blur_img, Size(3, 3));

    Mat saddle = getSaddle(blur_img);
    saddle = -saddle;
    Mat temp = saddle < 0;
    saddle.setTo(0, temp);

    pruneSaddle(saddle);
    Mat s2 = nonmax_sup(saddle);
    temp = s2 < 100000;
    s2.setTo(0, temp);

    vector<Point> spts;
    for (int y = 0; y < s2.rows; y++) {
        for (int x = 0; x < s2.cols; x++) {
            if (s2.at<double>(y, x) != 0) {
                spts.push_back(Point(x, y));
            }
        }
    }
    // cout << "Number of saddle points: " << spts.size() << endl;

    Mat edges;
    Canny(img, edges, 20, 250);

    vector<vector<Point>> contours_all;
    vector<Vec4i> hierarchy_all;
    getContours(img, edges, contours_all, hierarchy_all);
    // cout << "Number of contours found: " << contours_all.size() << endl;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    pruneContours(contours_all, hierarchy_all, saddle, contours, hierarchy);
    // cout << "Number of contours after pruning: " << contours.size() << endl;

    int curr_num_good = 0;
    Mat curr_grid_next, curr_grid_good, curr_M;

    for (size_t cnt_i = 0; cnt_i < contours.size(); cnt_i++) {
        vector<Point> cnt = contours[cnt_i];

        Mat grid_curr, ideal_grid, M;
        tie(grid_curr, ideal_grid, M) = getInitChessGrid(cnt);

        int num_good = 0;
        Mat grid_next, grid_good_mask;

        for (int grid_i = 0; grid_i < 7; grid_i++) {
            Mat ignore;
            tie(grid_curr, ideal_grid, ignore) = makeChessGrid(M, grid_i + 1);

            tie(grid_next, grid_good_mask) = findGoodPoints(grid_curr, spts);
            num_good = countNonZero(grid_good_mask);

            if (num_good < 4) {
                M.release();
                break;
            }

            M = generateNewBestFit(ideal_grid, grid_next, grid_good_mask);

            if (M.empty() || abs(M.at<double>(0, 0) / M.at<double>(1, 1)) > 15 ||
                abs(M.at<double>(1, 1) / M.at<double>(0, 0)) > 15) {
                M.release();
                break;
            }
        }

        if (M.empty())
            continue;

        if (num_good > curr_num_good) {
            curr_num_good = num_good;
            curr_grid_next = grid_next.clone();
            curr_grid_good = grid_good_mask.clone();
            curr_M = M.clone();
        }

        if (num_good > max_pts_needed)
            break;
    }

    // cout << "Current number of good points: " << curr_num_good << endl;

    if (curr_num_good > min_pts_needed) {
        Mat final_ideal_grid = getIdentityGrid(2 + 2 * 7);
        final_ideal_grid -= Scalar(7, 7);
        final_ideal_grid.convertTo(final_ideal_grid, CV_32F);
        return make_tuple(curr_M, final_ideal_grid, curr_grid_next, curr_grid_good, spts);
    }
    else {
        return make_tuple(Mat(), Mat(), Mat(), Mat(), vector<Point>());
    }
}

Mat getUnwarpedPoints(const vector<int>& best_lines_x,
    const vector<int>& best_lines_y,
    const Mat& M) {
    vector<Point2f> xy;
    for (int y : best_lines_y) {
        for (int x : best_lines_x) {
            xy.push_back(Point2f(float(x), float(y)));
        }
    }

    Mat xy_mat(xy.size(), 1, CV_32FC2);
    for (size_t i = 0; i < xy.size(); i++) {
        xy_mat.at<Point2f>(i, 0) = xy[i];
    }

    Mat xy_unwarp;
    perspectiveTransform(xy_mat, xy_unwarp, M);
    xy_unwarp = xy_unwarp.reshape(1, xy_unwarp.rows);

    return xy_unwarp;
}

Mat getBoardOutline(const vector<int>& best_lines_x,
    const vector<int>& best_lines_y,
    const Mat& M) {
    int d = best_lines_x[1] - best_lines_x[0];
    vector<int> ax = {best_lines_x[0] - d, best_lines_x.back() + d};
    vector<int> ay = {best_lines_y[0] - d, best_lines_y.back() + d};

    vector<Point2f> xy;
    xy.push_back(Point2f(ax[0], ay[0]));
    xy.push_back(Point2f(ax[1], ay[0]));
    xy.push_back(Point2f(ax[1], ay[1]));
    xy.push_back(Point2f(ax[0], ay[1]));
    xy.push_back(Point2f(ax[0], ay[0]));

    Mat xy_mat(xy.size(), 1, CV_32FC2);
    for (size_t i = 0; i < xy.size(); i++) {
        xy_mat.at<Point2f>(i, 0) = xy[i];
    }

    Mat xy_unwarp;
    perspectiveTransform(xy_mat, xy_unwarp, M);
    xy_unwarp = xy_unwarp.reshape(1, xy_unwarp.rows);

    return xy_unwarp;
}

void processSingle(const string& filename) {
    Mat img = loadImage(filename);
    if (img.empty()) return;

    Mat M, ideal_grid, grid_next, grid_good;
    vector<Point> spts;

    tie(M, ideal_grid, grid_next, grid_good, spts) = findChessboard(img);

    if (!M.empty()) {
        Mat scaled_ideal = ((ideal_grid + 8) * 32);
        M = generateNewBestFit(scaled_ideal, grid_next, grid_good);

        Mat img_warp;
        warpPerspective(img, img_warp, M, Size(17 * 32, 17 * 32), WARP_INVERSE_MAP);

        vector<int> best_lines_x, best_lines_y;
        tie(best_lines_x, best_lines_y) = getBestLines(img_warp);

        Mat xy_unwarp_mat = getUnwarpedPoints(best_lines_x, best_lines_y, M);
        Mat board_outline_mat = getBoardOutline(best_lines_x, best_lines_y, M);

        Mat vis;
        cvtColor(img, vis, COLOR_GRAY2BGR);

        for (int i = 0; i < xy_unwarp_mat.rows; i++) {
            Point2f p(xy_unwarp_mat.at<float>(i, 0), xy_unwarp_mat.at<float>(i, 1));
            circle(vis, p, 3, Scalar(0, 255, 0), -1);
        }

        int nx = best_lines_x.size();
        int ny = best_lines_y.size();

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i + 1 < nx; i++) {
                Point2f p1(xy_unwarp_mat.at<float>(j * nx + i, 0), xy_unwarp_mat.at<float>(j * nx + i, 1));
                Point2f p2(xy_unwarp_mat.at<float>(j * nx + (i + 1), 0), xy_unwarp_mat.at<float>(j * nx + (i + 1), 1));
                line(vis, p1, p2, Scalar(0, 200, 255), 1);
            }
        }

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j + 1 < ny; j++) {
                Point2f p1(xy_unwarp_mat.at<float>(j * nx + i, 0), xy_unwarp_mat.at<float>(j * nx + i, 1));
                Point2f p2(xy_unwarp_mat.at<float>((j + 1) * nx + i, 0), xy_unwarp_mat.at<float>((j + 1) * nx + i, 1));
                line(vis, p1, p2, Scalar(255, 180, 0), 1);
            }
        }

        for (int i = 0; i < board_outline_mat.rows; i++) {
            Point2f p(board_outline_mat.at<float>(i, 0), board_outline_mat.at<float>(i, 1));
            circle(vis, p, 5, Scalar(0, 0, 255), -1);
        }

        imshow("Recovered Grid on Original Image", vis);
        waitKey(0);
    }
}

py::array_t<float> detectChessboardCorners(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img_array,
    const ChessboardDetectionConfig& config
) {
    try {
        // Convert numpy array to OpenCV Mat
        NumpyArrayWrapper<uint8_t> wrapper(img_array);
        Mat img = wrapper.toMat();
        
        // Convert to grayscale if needed
        if (img.channels() == 3) {
            cvtColor(img, img, COLOR_RGB2GRAY);
        } else if (img.channels() == 4) {
            cvtColor(img, img, COLOR_RGBA2GRAY);
        }
        
        // Resize if too large
        if (img.cols > config.max_image_size || img.rows > config.max_image_size) {
            double scale = min(
                config.max_image_size / double(img.cols),
                config.max_image_size / double(img.rows)
            );
            resize(img, img, Size(), scale, scale, INTER_LINEAR);
        }


        // Run the detection pipeline
        Mat M, ideal_grid, grid_next, grid_good;
        vector<Point> spts;
        tie(M, ideal_grid, grid_next, grid_good, spts) = findChessboard(img);

        // Check if detection was successful
        if (M.empty()) {
            throw std::runtime_error("Chessboard detection failed");
        }

        // Get board outline corners
        Mat scaled_ideal = ((ideal_grid + 8) * 32);
        M = generateNewBestFit(scaled_ideal, grid_next, grid_good);

        Mat img_warp;
        warpPerspective(img, img_warp, M, Size(17 * 32, 17 * 32), WARP_INVERSE_MAP);

        vector<int> best_lines_x, best_lines_y;
        tie(best_lines_x, best_lines_y) = getBestLines(img_warp);

        Mat board_outline_mat = getBoardOutline(best_lines_x, best_lines_y, M);

        // Extract the 4 corner points (first 4 rows of board_outline_mat)
        if (board_outline_mat.rows < 4) {
            throw std::runtime_error("Could not extract board outline corners");
        }

        // Create output numpy array (4, 2) with float32
        py::array_t<float> corners({4, 2});
        auto buf_out = corners.request();
        float* ptr_out = (float*)buf_out.ptr;

        for (int i = 0; i < 4; i++) {
            ptr_out[i * 2 + 0] = board_outline_mat.at<float>(i, 0);  // x coordinate
            ptr_out[i * 2 + 1] = board_outline_mat.at<float>(i, 1);  // y coordinate
        }

        return corners;
    } catch (const exception& e) {
        throw std::runtime_error(string("Chessboard detection failed: ") + e.what());
    }
}