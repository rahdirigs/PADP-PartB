#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <utility>

const int BLOCK_SIZE = 16;
const int FILTER_HEIGHT = 3;
const int FILTER_WIDTH = 3;

using namespace std;
using namespace cv;

pair<string, string> process_file_name(const string &input_image) {
  int len = (int) input_image.length();
  int dot = -1;
  for (int i = 0; i < len; ++i) {
    if (input_image[i] == '.') {
      dot = i;
      break;
    }
  }
  if (dot == -1) {
    cerr << "Invalid file name\n";
    exit(0);
  }
  string name = input_image.substr(0, dot);
  string ext = input_image.substr(dot + 1, len - dot - 1);
  return make_pair(name, ext);
}

__global__ void total_variation_filter(unsigned char *img, unsigned char *omg, unsigned int width, unsigned int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  bool filter_inside_image = (x >= FILTER_WIDTH / 2) && (x < width - FILTER_WIDTH / 2) && (y >= FILTER_HEIGHT / 2) && (y < height - FILTER_HEIGHT / 2);
  if (filter_inside_image) {
    float sum = 0.0;
    for (int i = -FILTER_HEIGHT / 2; i <= FILTER_HEIGHT / 2; ++i) {
      for (int j = -FILTER_WIDTH / 2; j <= FILTER_WIDTH / 2; ++j) {
        float value = img[(y + i) * width + (x + j)];
        float center = img[y * width + x];
        sum += (value - center);
      }
    }
    omg[y * width + x] = sum;
  }
}

void filter(Mat &img, Mat &omg) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  const int input_size = img.rows * img.cols;
  const int output_size = omg.rows * omg.cols;
  unsigned char *d_img, *d_omg;
  cudaMalloc<unsigned char>(&d_img, input_size);
  cudaMalloc<unsigned char>(&d_omg, output_size);
  cudaMemcpy(d_img, img.ptr(), input_size, cudaMemcpyHostToDevice);
  const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 grid((omg.cols + block.x - 1) / block.x, (omg.rows + block.y - 1) / block.y);
  cudaEventRecord(start);
  total_variation_filter<<<grid, block>>>(d_img, d_omg, omg.cols, omg.rows);
  cudaEventRecord(stop);
  cudaMemcpy(omg.ptr(), d_omg, output_size, cudaMemcpyDeviceToHost);
  cudaFree(d_img);
  cudaFree(d_omg);
  cudaEventSynchronize(stop);
  float time_elapsed = 0.0;
  cudaEventElapsedTime(&time_elapsed, start, stop);
  cout << "Time taken for execution = " << time_elapsed << "ms.\n";
}

int main(int argc, char **argv) {
  if (argc != 2) {
    cerr << "Usage: ./main <input_image>";
    return 0;
  }
  string input_image = string(argv[1]);
  pair<string, string> name_extension = process_file_name(input_image);
  string input_image_name = name_extension.first;
  string input_image_ext = name_extension.second;
  string output_image = input_image_name + "_out." + input_image_ext;
  Mat img = imread(input_image, CV_LOAD_IMAGE_UNCHANGED);
  if (img.empty()) {
    cerr << "No image found " << input_image << "\n";
    return 0;
  }
  cout << "Input image dimensions:\n";
  cout << "Rows = " << img.rows << "\nColumns = " << img.cols << "\n";
  cvtColor(img, img, CV_BGR2GRAY);
  Mat omg(img.size(), img.type());
  filter(img, omg);
  omg.convertTo(omg, CV_32F, 1.0 / 255, 0);
  omg *= 255;
  imwrite(output_image, omg);
  return 0;
}
