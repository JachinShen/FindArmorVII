#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Armor.h"
#include "Serial.h"

int main(void)
{
    cv::VideoCapture cap(1);
    //cv::VideoCapture cap("/home/jachinshen/视频/MArmor.avi"); 
    cv::Mat src;
    Armor armor;
    cv::namedWindow("frame", 1); 
    cap.read(src);
    cv::imshow("frame", src);
    cv::waitKey(0);

    Serial serial;

    armor.setDraw(SHOW_ALL);
    armor.init(src);
    serial.init();
    while(cap.read(src))	
    {
        armor.feedImage(src);
        //cout << "x:" << armor.getTargetX()
            //<< " y:" << armor.getTargetY() << endl;
        serial.sendTarget(armor.getTargetX(), armor.getTargetY(), armor.isFound());

        cv::imshow("frame", src);
        cv::waitKey(1);
    }
    cap.release();
}
