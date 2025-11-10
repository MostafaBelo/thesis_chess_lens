#include "PieceCropper.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <opencv2/opencv.hpp>


namespace {
    template <typename T>
    class NumpyArray3D {
        private:
            struct Index3D {
                ssize_t i,j,k;
            };
        public:
            NumpyArray3D(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
                info = arr.request();
                if (info.ndim != 3) throw std::runtime_error("Expected a 3D NumPy array of shape");
                ptr = static_cast<T*>(info.ptr);
                shape = {
                    info.shape[0],
                    info.shape[1],
                    info.shape[2],
                };
            }

            T& operator[](Index3D index) {
                return at(index);
            }

            string str() {
                string res = "[";
                for (int i = 0; i < 8; i++)
                {
                    res += "[";
                    for (int j = 0; j < 8; j++)
                    {
                        res += "[";
                        for (int k = 0; k < 13; k++)
                        {
                            res += to_string(at({i,j,k}));
                            if (k != 12) res += ", ";
                        }
                        res += "]";
                        if (j != 7) res += "\n";
                    }
                    res += "]";
                    if (i != 7) res += "\n\n";
                }
                res += "]";
                return res;
            }

            Index3D shape;
        private:
            py::buffer_info info;
            T* ptr;

            T& at(Index3D index) {
                ssize_t flat_index = index.i * shape.j * shape.k +
                                    index.j * shape.k +
                                    index.k;
                return ptr[flat_index];
            }
    };


    template <typename T>
    class NumpyArray4D {
        private:
            struct Index4D {
                ssize_t i,j,k,w;
            };
        public:
            NumpyArray4D(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
                info = arr.request();
                if (info.ndim != 4) throw std::runtime_error("Expected a 4D NumPy array of shape");
                ptr = static_cast<T*>(info.ptr);
                shape = {
                    info.shape[0],
                    info.shape[1],
                    info.shape[2],
                    info.shape[3]
                };
            }

            T& operator[](Index4D index) {
                return at(index);
            }

            string str() {
                string res = "[";
                for (int i = 0; i < shape.i; i++)
                {
                    res += "[";
                    for (int j = 0; j < shape.j; j++)
                    {
                        res += "[";
                        for (int k = 0; k < shape.k; k++)
                        {
                            res += "[";
                            for (int w = 0; w < shape.w; w++)
                            {
                                res += to_string(at({i,j,k,w}));
                                if (w != shape.w-1) res += ", ";
                            }
                            res += "]";
                            if (k != shape.k-1) res += "\n";
                        }
                        res += "]";
                        if (j != shape.j-1) res += "\n\n";
                    }
                    res += "]";
                    if (i != shape.i-1) res += "\n\n\n";
                }
                res += "]";
                return res;
            }

            Index4D shape;
        private:
            py::buffer_info info;
            T* ptr;

            T& at(Index4D index) {
                ssize_t flat_index = index.i * shape.j * shape.k * shape.w +
                                    index.j * shape.k * shape.w +
                                    index.k * shape.w +
                                    index.w;
                return ptr[flat_index];
            }
    };

    template <typename T>
    class NumpyArray5D {
        private:
            struct Index5D {
                ssize_t i,j,k,w,v;
            };
        public:
            NumpyArray5D(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
                info = arr.request();
                if (info.ndim != 5) throw std::runtime_error("Expected a 5D NumPy array of shape");
                ptr = static_cast<T*>(info.ptr);
                shape = {
                    info.shape[0],
                    info.shape[1],
                    info.shape[2],
                    info.shape[3],
                    info.shape[4]
                };
            }

            T& operator[](Index5D index) {
                return at(index);
            }

            string str() {
                string res = "[";
                for (int i = 0; i < shape.i; i++)
                {
                    res += "[";
                    for (int j = 0; j < shape.j; j++)
                    {
                        res += "[";
                        for (int k = 0; k < shape.k; k++)
                        {
                            res += "[";
                            for (int w = 0; w < shape.w; w++)
                            {
                                res += "[";
                                for (int v = 0; v < shape.v; v++)
                                {
                                    res += to_string(at({i,j,k,w,v}));
                                    if (v != shape.v-1) res += ", ";
                                }
                                res += "]";
                                if (w != shape.w-1) res += "\n";
                            }
                            res += "]";
                            if (k != shape.k-1) res += "\n\n";
                        }
                        res += "]";
                        if (j != shape.j-1) res += "\n\n\n";
                    }
                    res += "]";
                    if (i != shape.i-1) res += "\n\n\n\n";
                }
                res += "]";
                return res;
            }

            Index5D shape;
        private:
            py::buffer_info info;
            T* ptr;

            T& at(Index5D index) {
                ssize_t flat_index = index.i * shape.j * shape.k * shape.w * shape.v +
                                    index.j * shape.k * shape.w * shape.v +
                                    index.k * shape.w * shape.v +
                                    index.w * shape.v +
                                    index.v;
                return ptr[flat_index];
            }
    };
}


py::array_t<uint8_t, py::array::c_style | py::array::forcecast> extract_warped_squares(
    py::array_t<float, py::array::c_style | py::array::forcecast> grid_top,
    py::array_t<float, py::array::c_style | py::array::forcecast> grid_bottom,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img_numpy,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> res
) {
    NumpyArray3D grid_top_arr(grid_top);
    NumpyArray3D grid_bottom_arr(grid_bottom);
    NumpyArray3D img_numpy_arr(img_numpy);
    NumpyArray5D res_arr(res);

    int sq_height = 128;
    int sq_width = 64;

    if (grid_top_arr.shape.i != 9 || grid_top_arr.shape.j != 9 || grid_top_arr.shape.k != 2) throw invalid_argument("Numpy Shape is Invalid");
    if (grid_bottom_arr.shape.i != 9 || grid_bottom_arr.shape.j != 9 || grid_bottom_arr.shape.k != 2) throw invalid_argument("Numpy Shape is Invalid");

    ssize_t H = img_numpy_arr.shape.i;
    ssize_t W = img_numpy_arr.shape.j;
    if (img_numpy_arr.shape.k != 3) throw invalid_argument("Numpy Shape is Invalid");


    // cv::Mat img(H, W, CV_8UC3, (void*)img_numpy.data());
    cv::Mat img(H, W, CV_8UC3, (void*)(&(img_numpy_arr[{0,0,0}])));

    cv::Point2f dst_corners[4] = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(sq_width), 0),
        cv::Point2f(static_cast<float>(sq_width), static_cast<float>(sq_height)),
        cv::Point2f(0, static_cast<float>(sq_height))
    };

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            // --- extract top and bottom corners ---
            // cv::Point2f crop_corners[4];
            // for (int i = 0; i < 2; ++i) {
            //     for (int j = 0; j < 2; ++j) {
            //         int rr_top = r + i;
            //         int cc_top = c + j;
            //         int rr_bottom = r + i;
            //         int cc_bottom = c + j;

            //         if (i == 0) { // top row
            //             float top_x = grid_top_arr[{rr_top, cc_top, 0}];
            //             float top_y = grid_top_arr[{rr_top, cc_top, 1}];
            //             crop_corners[i*2 + j] = cv::Point2f(
            //                 top_x, top_y
            //             );
            //         } else { // bottom row
            //             float bottom_x = grid_bottom_arr[{rr_bottom, cc_bottom, 0}];
            //             float bottom_y = grid_bottom_arr[{rr_bottom, cc_bottom, 1}];
            //             crop_corners[i*2 + j] = cv::Point2f(
            //                 bottom_x, bottom_y
            //             );
            //         }
            //     }
            // }
            float x_min_top = std::min(grid_top_arr[{r, c, 0}], grid_top_arr[{r, c+1, 0}]);
            float x_max_top = std::max(grid_top_arr[{r, c, 0}], grid_top_arr[{r, c+1, 0}]);
            float y_min_top = std::min(grid_top_arr[{r, c, 1}], grid_top_arr[{r, c+1, 1}]);

            float x_min_bottom = std::min(grid_bottom_arr[{r+1, c, 0}], grid_bottom_arr[{r+1, c+1, 0}]);
            float x_max_bottom = std::max(grid_bottom_arr[{r+1, c, 0}], grid_bottom_arr[{r+1, c+1, 0}]);
            float y_max_bottom = std::max(grid_bottom_arr[{r+1, c, 1}], grid_bottom_arr[{r+1, c+1, 1}]);

            cv::Point2f crop_corners[4] = {
                cv::Point2f(x_min_top, y_min_top),
                cv::Point2f(x_max_top, y_min_top),
                cv::Point2f(x_max_bottom, y_max_bottom),
                cv::Point2f(x_min_bottom, y_max_bottom)
            };

            // --- 6. Get perspective transform ---
            cv::Mat M = cv::getPerspectiveTransform(crop_corners, dst_corners);

            // --- 7. Warp image ---
            cv::Mat warped;
            cv::warpPerspective(img, warped, M, cv::Size(sq_width, sq_height));

            cv::Point2f bottom_corners[4] = {
                cv::Point2f(grid_bottom_arr[{r,c,0}], grid_bottom_arr[{r,c,1}]),
                cv::Point2f(grid_bottom_arr[{r,c+1,0}], grid_bottom_arr[{r,c+1,1}]),
                cv::Point2f(grid_bottom_arr[{r+1,c+1,0}], grid_bottom_arr[{r+1,c+1,1}]),
                cv::Point2f(grid_bottom_arr[{r+1,c,0}], grid_bottom_arr[{r+1,c,1}])
            };

            std::vector<cv::Point2f> bottom_corners_vec(bottom_corners, bottom_corners + 4);
            std::vector<cv::Point2f> warped_corners_vec;
            cv::perspectiveTransform(bottom_corners_vec, warped_corners_vec, M);

            // --- Draw blue dots at each warped bottom corner ---
            for (const auto &p : warped_corners_vec) {
                int lower_y = std::max(static_cast<int>(p.y) - 5, 0);
                int upper_y = std::min(static_cast<int>(p.y) + 5, sq_height);
                int lower_x = std::max(static_cast<int>(p.x) - 5, 0);
                int upper_x = std::min(static_cast<int>(p.x) + 5, sq_width);

                for (int yy = lower_y; yy < upper_y; ++yy) {
                    for (int xx = lower_x; xx < upper_x; ++xx) {
                        warped.at<cv::Vec3b>(yy, xx)[0] = 0; // Blue
                        warped.at<cv::Vec3b>(yy, xx)[1] = 0;
                        warped.at<cv::Vec3b>(yy, xx)[2] = 255;
                    }
                }
            }

            // --- 8. Copy to output buffer ---
            for (int y = 0; y < sq_height; ++y) {
                for (int x = 0; x < sq_width; ++x) {
                    cv::Vec3b pixel = warped.at<cv::Vec3b>(y, x);
                    res_arr[{r,c,y,x,0}] = pixel[0];
                    res_arr[{r,c,y,x,1}] = pixel[1];
                    res_arr[{r,c,y,x,2}] = pixel[2];
                }
            }
        }
    }

    return res;
}