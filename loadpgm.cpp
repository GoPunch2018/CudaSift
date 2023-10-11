#include <opencv2/opencv.hpp>

int main() {
    // 图片文件名列表
    std::vector<std::string> imageFiles = {"data/img1.png", "data/img2.png", "data/left.pgm","data/righ.pgm","data/rimg_pts.pgm"};

    // 创建多个窗口
    std::vector<std::string> windowNames = {"Window1", "Window2", "Window3","Window4","Window5"};
    for (const std::string& windowName : windowNames) {
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    }

    // 逐个读取图片并显示在对应的窗口中
    for (int i = 0; i < imageFiles.size(); i++) {
        // 读取图片
        cv::Mat image = cv::imread(imageFiles[i]);

        // 检查图片是否成功读取
        if (image.empty()) {
            std::cout << "无法打开图像文件: " << imageFiles[i] << std::endl;
            continue;
        }

        // 显示图片在对应的窗口中
        cv::imshow(windowNames[i], image);
    }

    // 等待用户按下任意键后关闭窗口
    cv::waitKey(0);

    return 0;
}
