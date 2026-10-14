#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 如果连上了摄像头用 cap(0)，否则先用 cap("test.mp4")
    VideoCapture cap(0); 
    if (!cap.isOpened()) {
        cerr << "错误：摄像头或视频源打开失败！" << endl;
        return -1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    Mat frame, hsv, maskRed, maskBlue;
    cout << "程序启动，按 ESC 退出" << endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, hsv, COLOR_BGR2HSV);

        // 1. 提取目标（红色）
        Mat maskR1, maskR2;
        inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), maskR1);
        inRange(hsv, Scalar(160, 100, 100), Scalar(180, 255, 255), maskR2);
        maskRed = maskR1 | maskR2;
        
        // 2. 提取高地区域（假设为蓝色，根据实际场地调整）
        inRange(hsv, Scalar(100, 100, 100), Scalar(124, 255, 255), maskBlue);

        // 形态学去噪
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(maskRed, maskRed, MORPH_OPEN, kernel);
        morphologyEx(maskBlue, maskBlue, MORPH_OPEN, kernel);

        // 3. 找高地区域（找最大的蓝色区域）
        vector<vector<Point>> blueContours;
        findContours(maskBlue, blueContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        Rect highZoneRect; 
        bool hasHighZone = false;
        double maxBlueArea = 0;
        
        for (const auto& cnt : blueContours) {
            double area = contourArea(cnt);
            if (area > 2000 && area > maxBlueArea) { 
                maxBlueArea = area;
                highZoneRect = boundingRect(cnt);
                hasHighZone = true;
            }
        }

        if (hasHighZone) {
            rectangle(frame, highZoneRect, Scalar(255, 0, 0), 2); // 画蓝色高地框
            putText(frame, "High Zone", Point(highZoneRect.x, highZoneRect.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
        }

        // 4. 找目标并判断
        vector<vector<Point>> redContours;
        findContours(maskRed, redContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : redContours) {
            double area = contourArea(cnt);
            if (area < 300 || area > 50000) continue; // 面积过滤

            Rect targetRect = boundingRect(cnt);
            rectangle(frame, targetRect, Scalar(0, 255, 0), 2); // 画绿色目标框

            Point targetCenter(targetRect.x + targetRect.width / 2, targetRect.y + targetRect.height / 2);
            circle(frame, targetCenter, 5, Scalar(0, 255, 255), -1); 

            // 核心判定：判断目标中心点是否在蓝色高地框内
            if (hasHighZone && highZoneRect.contains(targetCenter)) {
                putText(frame, "IN HIGH ZONE!", Point(targetCenter.x - 50, targetCenter.y + 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                cout << ">>> 判定：目标已进入高分区！ <<<" << endl;
            } else {
                cout << "目标不在高分区..." << endl;
            }
        }

        imshow("High Zone Detection", frame);
        imshow("Red Mask", maskRed);   // 调参必看
        imshow("Blue Mask", maskBlue); // 调参必看

        if (waitKey(30) == 27) break; // ESC退出
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
