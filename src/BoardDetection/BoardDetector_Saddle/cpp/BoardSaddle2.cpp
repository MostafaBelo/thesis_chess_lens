// #include "BoardSaddle.h"
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
        // Create blank image if none provided
        vis = Mat::zeros(600, 600, CV_8UC3);
    }

    // 1. Draw original ideal grid (blue)
    for (int i = 0; i < ideal_grid.rows; i++) {
        circle(vis, Point2f(ideal_grid.at<float>(i, 0), ideal_grid.at<float>(i, 1)), 4, Scalar(255, 0, 0), -1);
    }

    // 2. Draw warped grid (green)
    for (int i = 0; i < warped_grid.rows; i++) {
        circle(vis, Point2f(warped_grid.at<float>(i, 0), warped_grid.at<float>(i, 1)), 4, Scalar(0, 255, 0), -1);
    }

    // 3. Draw good points (red)
    for (int i = 0; i < new_grid.rows; i++) {
        bool is_good = false;
        if (grid_good.type() == CV_8U)
            is_good = grid_good.at<uchar>(i, 0) > 0;
        else if (grid_good.type() == CV_8UC1) // optional fallback
            is_good = grid_good.at<uchar>(i, 0) > 0;

        if (is_good) {
            circle(vis, Point2f(new_grid.at<float>(i, 0), new_grid.at<float>(i, 1)), 4, Scalar(0, 0, 255), -1);
        }
    }

    // 4. Draw detected saddle points (yellow)
    for (const auto& pt : spts) {
        circle(vis, pt, 3, Scalar(0, 255, 255), -1);
    }

    // 5. Show result
    imshow("Chess Grid Visualization", vis);
    waitKey(0);

    // Optional: Print numeric comparison
    for (int i = 0; i < new_grid.rows; i++) {
        cout << "Pt " << i
            << ": ideal=(" << ideal_grid.at<float>(i, 0) << "," << ideal_grid.at<float>(i, 1) << ")"
            << " warped=(" << warped_grid.at<float>(i, 0) << "," << warped_grid.at<float>(i, 1) << ")"
            << " good=(" << new_grid.at<float>(i, 0) << "," << new_grid.at<float>(i, 1) << ")"
            << " good_flag=" << (int)grid_good.at<uchar>(i, 0)
            << endl;
    }
}


vector<Point> updateCorners(const vector<Point>& contour, const Mat& saddle) {
    int ws = 4;
    vector<Point> new_contour = contour;

    for (size_t i = 0; i < contour.size(); i++) {
        int cc = contour[i].x;
        int rr = contour[i].y;

        int rl = max(0, rr - ws);
        int cl = max(0, cc - ws);
        int br, bc;

        Mat window = saddle(Range(rl, min(saddle.rows, rr + ws + 1)),
            Range(cl, min(saddle.cols, cc + ws + 1)));

        double minVal, maxVal;
        Point maxLoc;
        minMaxLoc(window, &minVal, &maxVal, nullptr, &maxLoc);
        br = maxLoc.y - min(ws, rl);
        bc = maxLoc.x - min(ws, cl);

        if (maxVal > 0)
            new_contour[i] = Point(cc + bc, rr + br);
        else
            return {}; // empty vector if invalid
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
    double xratio = (xa < xb) ? xa / xb : xb / xa;

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

// ---------------- Core Functions ----------------
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
    for (auto& cnt : contours) {
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
    findContours(edges_gradient, tmp_contours, tmp_hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

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
        if (contourArea(cnt) < 64) continue;  // 8x8 minimum area
        if (!is_square(cnt)) continue;

        // Here you would call updateCorners if implemented
        new_contours.push_back(cnt);
        new_hierarchy.push_back(h);
    }

    if (new_contours.empty()) return;

    vector<double> areas;
    for (auto& cnt : new_contours)
        areas.push_back(contourArea(cnt));

    double median_area;
    sort(areas.begin(), areas.end());
    median_area = areas[areas.size() / 2];

    vector<vector<Point>> filtered_contours;
    vector<Vec4i> filtered_hierarchy;
    for (size_t i = 0; i < new_contours.size(); i++) {
        if (areas[i] >= 0.25 * median_area && areas[i] <= 2 * median_area) {
            filtered_contours.push_back(new_contours[i]);
            filtered_hierarchy.push_back(new_hierarchy[i]);
        }
    }

    new_contours = filtered_contours;
    new_hierarchy = filtered_hierarchy;
}






//Mat getInitChessGrid(const Mat& quad) {
//    Mat quadA = (Mat_<float>(4, 2) << 0, 1, 1, 1, 1, 0, 0, 0);
//    Mat M = getPerspectiveTransform(quadA, quad);
//    auto [grid, _, __] = makeChessGrid(M, 1);
//    return grid;
//}
//



// ---------------- Identity Grid ----------------
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

// ---------------- Chess Grid ----------------
Mat getChessGrid(const Mat& quad) {
    Mat quadA = (Mat_<float>(4, 2) << 0, 1, 1, 1, 1, 0, 0, 0);
    Mat M = getPerspectiveTransform(quadA, quad);
    Mat quadB = getIdentityGrid(4) - 1;

    Mat quadB_pad;
    hconcat(quadB, Mat::ones(quadB.rows, 1, CV_32F), quadB_pad); // pad with 1
    Mat C_thing = M * quadB_pad.t();  // 3xN
    Mat C_thing_t = C_thing.t();
    for (int i = 0; i < C_thing_t.rows; i++) {
        C_thing_t.at<float>(i, 0) /= C_thing_t.at<float>(i, 2);
        C_thing_t.at<float>(i, 1) /= C_thing_t.at<float>(i, 2);
    }
    return C_thing_t.colRange(0, 2);
}

// ---------------- Good Points ----------------
tuple<Mat, Mat> findGoodPoints(const Mat& grid, const vector<Point>& spts, double max_px_dist = 5.0) {
    Mat new_grid = grid.clone();
    Mat grid_good = Mat::zeros(grid.rows, 1, CV_8U);
    set<string> chosen_spts;

    auto hash_pt = [](const Point2f& pt) { return to_string(int(pt.x)) + "_" + to_string(int(pt.y)); };

    for (int i = 0; i < grid.rows; i++) {
        Point2f pt(grid.at<float>(i, 0), grid.at<float>(i, 1));
        Point2f best_pt; double best_dist;

        // getMinSaddleDist equivalent
        best_dist = 1e9;
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

        string h = hash_pt(best_pt);
        if (chosen_spts.count(h)) best_dist = max_px_dist;
        else chosen_spts.insert(h);

        if (best_dist < max_px_dist) {
            new_grid.at<float>(i, 0) = best_pt.x;
            new_grid.at<float>(i, 1) = best_pt.y;
            grid_good.at<uchar>(i, 0) = 1;
        }
    }
    return make_tuple(new_grid, grid_good);
}



tuple<Mat, Mat, Mat> makeChessGrid(const Mat& M, int N = 1)
{
    // 1. Make ideal grid as float32 and center it
    Mat ideal_grid = getIdentityGrid(2 + 2 * N);  // (K x 2)
    ideal_grid.convertTo(ideal_grid, CV_32F);
    ideal_grid -= Scalar(N, N);   // same as Python: ideal_grid - N

    // 2. Pad with a column of 1s → (K x 3)
    Mat ones_col = Mat::ones(ideal_grid.rows, 1, CV_32F);
    Mat ideal_grid_pad;
    hconcat(ideal_grid, ones_col, ideal_grid_pad);  // (K x 3)

    // 3. Convert M to CV_32F to match ideal_grid_pad
    Mat M32;
    M.convertTo(M32, CV_32F);

    // 4. Multiply homography
    Mat grid_h = M32 * ideal_grid_pad.t();  // (3 x K)
    Mat grid_t = grid_h.t();                // (K x 3)

    // 5. Normalize homogeneous coordinates
    Mat grid(grid_t.rows, 2, CV_32F);
    for (int i = 0; i < grid_t.rows; i++) {
        float xh = grid_t.at<float>(i, 0);
        float yh = grid_t.at<float>(i, 1);
        float wh = grid_t.at<float>(i, 2);

        grid.at<float>(i, 0) = xh / wh;
        grid.at<float>(i, 1) = yh / wh;
    }

    // 6. Optional debug
    // cout << "M type: " << M.type() << ", ideal_grid_pad type: " << ideal_grid_pad.type() << endl;

    // 7. Return same order as Python: (grid, ideal_grid, M)
    return make_tuple(grid, ideal_grid, M.clone());
}

// ---------------- Init Chess Grid ----------------
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


// ---------------- New Best Fit ----------------
Mat generateNewBestFit(const Mat& grid_ideal, const Mat& grid, const Mat& grid_good) {
    vector<Point2f> a, b;
    for (int i = 0; i < grid.rows; i++) {
        if (grid_good.at<uchar>(i, 0)) {
            a.push_back(Point2f(grid_ideal.at<float>(i, 0), grid_ideal.at<float>(i, 1)));
            b.push_back(Point2f(grid.at<float>(i, 0), grid.at<float>(i, 1)));
        }
    }
    Mat M = findHomography(a, b, RANSAC);
    return M;
}

// ---------------- Gradients ----------------
tuple<Mat, Mat, Mat, Mat, Mat> getGrads(const Mat& img) {
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
    return make_tuple(grad_mag, grad_phase_masked, grad_phase, gx, gy);
}

// ---------------- Best Lines ----------------
tuple<vector<int>, vector<int>> getBestLines(const Mat& img_warped) {
    Mat grad_mag, grad_phase_masked, grad_phase, gx, gy;
    tie(grad_mag, grad_phase_masked, grad_phase, gx, gy) = getGrads(img_warped);

    Mat gx_pos = gx.clone(); gx_pos.setTo(0, gx_pos < 0);
    Mat gx_neg = -gx; gx_neg.setTo(0, gx_neg < 0);
    Mat gy_pos = gy.clone(); gy_pos.setTo(0, gy_pos < 0);
    Mat gy_neg = -gy; gy_neg.setTo(0, gy_neg < 0);

    Mat score_x = Mat::zeros(1, gx.cols, CV_64F);
    Mat score_y = Mat::zeros(gy.rows, 1, CV_64F);

    for (int i = 0; i < gx.cols; i++)
        score_x.at<double>(0, i) = sum(gx_pos.col(i))[0] * sum(gx_neg.col(i))[0];
    for (int i = 0; i < gy.rows; i++)
        score_y.at<double>(i, 0) = sum(gy_pos.row(i))[0] * sum(gy_neg.row(i))[0];

    // Choose best internal set of 7 lines
    vector<vector<int>> a;
    for (int offset = 1; offset <= 8; offset++) {
        vector<int> pts(7);
        for (int j = 0; j < 7; j++) pts[j] = (offset + j + 1) * 32;
        a.push_back(pts);
    }
    vector<double> scores_x(a.size(), 0), scores_y(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++) {
        for (auto idx : a[i]) scores_x[i] += score_x.at<double>(0, idx);
        for (auto idx : a[i]) scores_y[i] += score_y.at<double>(idx, 0);
    }
    auto best_x = max_element(scores_x.begin(), scores_x.end()) - scores_x.begin();
    auto best_y = max_element(scores_y.begin(), scores_y.end()) - scores_y.begin();
    return make_tuple(a[best_x], a[best_y]);
}


// ------------------ Load Image ------------------
Mat loadImage(const string& filepath) {
    Mat img_orig = imread(filepath, IMREAD_COLOR);
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

    Mat img_gray;
    cvtColor(img_resized, img_gray, COLOR_BGR2GRAY);

    return img_gray;
}


tuple<Mat, Mat, Mat, Mat, vector<Point>> findChessboard(const Mat& img, int min_pts_needed = 15, int max_pts_needed = 25) {
    // Blur the image
    Mat blur_img;
    blur(img, blur_img, Size(3, 3));

    // Compute saddle points
    Mat saddle = getSaddle(blur_img);
    saddle = -saddle;
    threshold(saddle, saddle, 0, 0, THRESH_TOZERO);  // Set negative values to 0

    // Prune saddle points
    pruneSaddle(saddle);

    // Non-maximum suppression
    Mat s2 = nonmax_sup(saddle);
    threshold(s2, s2, 100000, 0, THRESH_TOZERO); // Keep only large values

    // Get saddle points positions
    vector<Point> spts;
    for (int y = 0; y < s2.rows; y++) {
        for (int x = 0; x < s2.cols; x++) {
            if (s2.at<double>(y, x) != 0) {  // <-- use double here
                spts.push_back(Point(x, y));
            }
        }
    }
    cout << endl << endl << "Number of saddle points: " << spts.size() << endl;



    // Edge detection
    Mat edges;
    Canny(img, edges, 20, 250);

    // Get and prune contours
    vector<vector<Point>> contours_all;
    vector<Vec4i> hierarchy_all;
    getContours(img, edges, contours_all, hierarchy_all);
    cout << endl << endl << "Number of contours found: " << contours_all.size() << endl;

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    pruneContours(contours_all, hierarchy_all, saddle, contours, hierarchy);
    cout << "Number of contours after pruning: " << contours.size() << endl << endl << endl;

    // ---------------->part2<------------
    int curr_num_good = 0;
    Mat curr_grid_next, curr_grid_good, curr_M;


    // Loop over contours
    for (size_t cnt_i = 0; cnt_i < contours.size(); cnt_i++) {
        vector<Point> cnt = contours[cnt_i];

        Mat grid_curr, ideal_grid, M;
        tie(grid_curr, ideal_grid, M) = getInitChessGrid(cnt);
        //cout << "--- getInitChessGrid output ---" << endl;

        //// Print first 5 points of grid_curr
        //cout << "grid_curr (first 5 points):" << endl;
        //for (int i = 0; i < min(5, grid_curr.rows); i++)
        //    cout << grid_curr.at<float>(i, 0) << ", " << grid_curr.at<float>(i, 1) << endl;

        //// Print first 5 points of ideal_grid
        //cout << "ideal_grid (first 5 points):" << endl;
        //for (int i = 0; i < min(5, ideal_grid.rows); i++)
        //    cout << ideal_grid.at<float>(i, 0) << ", " << ideal_grid.at<float>(i, 1) << endl;

        //// Print the homography matrix M
        //cout << "Homography M = " << endl << M << endl;

        //// Optional: print types
        //cout << "grid_curr type: " << grid_curr.type()
        //    << ", ideal_grid type: " << ideal_grid.type()
        //    << ", M type: " << M.type() << endl;
        int num_good = 0;
        
        // Declare these **outside** inner loop
        Mat grid_next, grid_good_mask;

        for (int grid_i = 0; grid_i < 7; grid_i++) {
            Mat ignore;
            tie(grid_curr, ideal_grid, ignore) = makeChessGrid(M, grid_i + 1);
            cout << "First 5 points in grid_curr:" << endl;
            for (int i = 0; i < min(5, grid_curr.rows); i++)
                cout << grid_curr.at<float>(i, 0) << ", " << grid_curr.at<float>(i, 1) << endl;

            // Call findGoodPoints
            tie(grid_next, grid_good_mask) = findGoodPoints(grid_curr, spts);

            num_good = countNonZero(grid_good_mask);
            if (num_good < 4) {
                M.release();
                break;
            }

            // Call generateNewBestFit
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


    cout << "currner_num_good" << curr_num_good;
    // Return final results
    if (curr_num_good > min_pts_needed) {
        Mat final_ideal_grid = getIdentityGrid(2 + 2 * 7) - 7;
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
    xy_unwarp = xy_unwarp.reshape(1, xy_unwarp.rows);  // Nx2

    return xy_unwarp;  // each row is (x, y)
}

void processSingle(const string& filename) {
    Mat img = loadImage(filename);
    if (img.empty()) return;

    Mat M, ideal_grid, grid_next;
    Mat grid_good;
    vector<Point> spts;

    tie(M, ideal_grid, grid_next, grid_good, spts) = findChessboard(img);
    
    cout << "we have reachhed here safely";

    if (!M.empty()) {
        M = generateNewBestFit((ideal_grid + 8) * 32, grid_next, grid_good);
        Mat img_warp;
        warpPerspective(img, img_warp, M, Size(17 * 32, 17 * 32), WARP_INVERSE_MAP);

        vector<int> best_lines_x, best_lines_y;
        tie(best_lines_x, best_lines_y) = getBestLines(img_warp);

        vector<Point2f> xy_unwarp = getUnwarpedPoints(best_lines_x, best_lines_y, M);

        Mat vis = img.clone();

        // Draw each recovered point on the visualization image
        for (const Point2f& p : xy_unwarp) {
            circle(vis, p, 3, Scalar(0, 255, 0), -1);  // small green point
        }

        // Optionally draw connecting lines to show grid structure:
        int nx = best_lines_x.size();
        int ny = best_lines_y.size();

        // Draw horizontal lines (across columns)
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i + 1 < nx; i++) {
                Point2f p1 = xy_unwarp[j * nx + i];
                Point2f p2 = xy_unwarp[j * nx + (i + 1)];
                line(vis, p1, p2, Scalar(0, 200, 255), 1);
            }
        }

        // Draw vertical lines (across rows)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j + 1 < ny; j++) {
                Point2f p1 = xy_unwarp[j * nx + i];
                Point2f p2 = xy_unwarp[(j + 1) * nx + i];
                line(vis, p1, p2, Scalar(255, 180, 0), 1);
            }
        }

        // Show the result
        imshow("Recovered Grid on Original Image", vis);
        waitKey(0);

    }
}

int main() {
    string path = "/mnt/D/University/Thesis_Dataset/Dataset/Images/image_173.png";
    try {
        processSingle(path);
    } catch (const exception& e) {
        cerr << e.what() << endl;
    }
    return 0;
}