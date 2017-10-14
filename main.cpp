#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <iostream>
#include <vector>
using namespace std;

#define DRAW NO_SHOW
#define SHOW_ALL 1
#define NO_SHOW 0
#define PI 3.1415926535898
#define V_INDEX 2
#define S_INDEX 1
#define H_INDEX 0
#define U8_MAX 255

class Armor
{
    private:
        int AREA_MAX;
        int AREA_MIN;
        int ERODE_KSIZE;
        int DILATE_KSIZE;
        int V_THRESHOLD;
        int S_THRESHOLD;
        int BORDER_THRESHOLD;
        int H_BLUE_LOW_THRESHOLD;
        int H_BLUE_LOW_THRESHOLD_MIN;
        int H_BLUE_LOW_THRESHOLD_MAX;
        int H_BLUE_HIGH_THRESHOLD;
        int H_BLUE_HIGH_THRESHOLD_MAX;
        int H_BLUE_HIGH_THRESHOLD_MIN;
        int H_BLUE_STEP;
        int H_BLUE_CHANGE_THRESHOLD_LOW;
        int H_BLUE_CHANGE_THRESHOLD_HIGH;
        int S_BLUE_THRESHOLD;
        int BLUE_PIXEL_RATIO_THRESHOLD;
        int R_MAX_THRESHOLD;
        int R_MIN_THRESHOLD;
        int FLOOD_LOWER;
        int FLOOD_UPPER;
        int CIRCLE_ROI_WIDTH;
        int CIRCLE_ROI_HEIGHT;
        int CIRCLE_THRESHOLD;
        

    private:
        cv::Mat hsv;
        cv::Mat s_low;
        cv::Mat s_canny;
        cv::Mat v_very_high;
        cv::Mat gray;
#if DRAW == SHOW_ALL
        cv::Mat light_draw;
#endif

        std::vector<cv::Mat > hsvSplit;
        std::vector<cv::RotatedRect> lights;
        std::vector<cv::Point > armors;
        std::vector<std::vector<cv::Point > > V_contours;

        cv::Mat V_element_erode;
        cv::Mat V_element_dilate;

        cv::RotatedRect rotated_rect;
        int srcH, srcW;
        cv::Point target;

    private:
        void getSrcSize(cv::Mat& src);
        void cvtHSV(cv::Mat& src);
        void cvtGray(cv::Mat& src);
        void getLightRegion();
        void getLightRegion2();
        void selectContours();
        bool isAreaTooBigOrSmall(int i);
        void getRotatedRect(int i);
        bool isCloseToBorder();
        bool isBlueNearby(int i);
        void pushLights();
#if DRAW == SHOW_ALL
        void drawLights();
#endif
        void selectLights();
        void chooseCloseTarget();
        bool isCircleAround(int midx, int midy);
        void cleanAll();

    public:
        Armor();
        void init();
        void feedImage(cv::Mat& src);
        bool isFound();
        int getTargetX();
        int getTargetY();
};

Armor::Armor():
    AREA_MAX(200),
    AREA_MIN(25),
    ERODE_KSIZE(2),
    DILATE_KSIZE(4),
    V_THRESHOLD(230),
    S_THRESHOLD(40),
    BORDER_THRESHOLD(10),
    H_BLUE_LOW_THRESHOLD(120),
    H_BLUE_LOW_THRESHOLD_MIN(100),
    H_BLUE_LOW_THRESHOLD_MAX(140),
    H_BLUE_HIGH_THRESHOLD(180),
    H_BLUE_HIGH_THRESHOLD_MAX(200),
    H_BLUE_HIGH_THRESHOLD_MIN(160),
    H_BLUE_STEP(1),
    H_BLUE_CHANGE_THRESHOLD_LOW(5),
    H_BLUE_CHANGE_THRESHOLD_HIGH(10),
    S_BLUE_THRESHOLD(100),
    BLUE_PIXEL_RATIO_THRESHOLD(12),
    R_MAX_THRESHOLD(15),
    R_MIN_THRESHOLD(5),
    FLOOD_LOWER(8),
    FLOOD_UPPER(8),
    CIRCLE_ROI_WIDTH(40),
    CIRCLE_ROI_HEIGHT(40),
    CIRCLE_THRESHOLD(80),
    target(Point(320,240))
{
}

void Armor::init()
{
    V_element_erode = cv::getStructuringElement(
            MORPH_CROSS, Size(ERODE_KSIZE, ERODE_KSIZE));
    V_element_dilate = cv::getStructuringElement(
            MORPH_CROSS, Size(DILATE_KSIZE, DILATE_KSIZE));
}

void Armor::feedImage(cv::Mat& src)
{
#if DRAW == SHOW_ALL
    light_draw = src.clone();
#endif
    cleanAll();
    getSrcSize(src);
    cvtHSV(src);
    cvtGray(src);
    getLightRegion();
    findContours(v_very_high, V_contours,
            CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    selectContours(); 
    selectLights();
    chooseCloseTarget();
}

void Armor::getSrcSize(cv::Mat& src)
{
    srcH = (int)src.size().height;
    srcW = (int)src.size().width;
}

void Armor::cvtHSV(cv::Mat& src)
{
    cv::cvtColor(src, hsv, CV_BGR2HSV_FULL);
    cv::split(hsv, hsvSplit);
}

void Armor::cvtGray(cv::Mat& src)
{
    cv::cvtColor(src, gray, CV_BGR2GRAY);
}
void Armor::getLightRegion()
{
    cv::threshold(hsvSplit[V_INDEX], v_very_high,
            V_THRESHOLD, U8_MAX, THRESH_BINARY);
    cv::threshold(hsvSplit[S_INDEX], s_low,
            S_THRESHOLD, U8_MAX, THRESH_BINARY_INV);
    bitwise_and(s_low, v_very_high, v_very_high);
    cv::erode(v_very_high, v_very_high, V_element_erode);
    cv::dilate(v_very_high, v_very_high, V_element_dilate);
#if DRAW == SHOW_ALL
    cv::namedWindow("light region",1);
    cv::imshow("light region", v_very_high);
    //cv::imshow("S region", s_low);
    cv::createTrackbar("V_THRESHOLD", "light region", &V_THRESHOLD, U8_MAX);
    cv::createTrackbar("S_THRESHOLD", "light region", &S_THRESHOLD, U8_MAX);
    cv::waitKey(1);
#endif
}

void Armor::getLightRegion2()
{
    cv::Canny(hsvSplit[S_INDEX], hsvSplit[S_INDEX], 400, 1000);
    cv::threshold(hsvSplit[V_INDEX], hsvSplit[V_INDEX],
            V_THRESHOLD, U8_MAX, THRESH_BINARY);
    cv::dilate(hsvSplit[V_INDEX], hsvSplit[V_INDEX], V_element_dilate);
    bitwise_and(hsvSplit[S_INDEX], hsvSplit[V_INDEX], hsvSplit[V_INDEX]);
#if DRAW == SHOW_ALL
    cv::namedWindow("light region",1);
    cv::imshow("light region", hsvSplit[V_INDEX]);
    cv::waitKey(1);
#endif

}

void Armor::selectContours()
{
        for (int i = 0; i < (int)V_contours.size(); i++)
        {
            if(isAreaTooBigOrSmall(i))
                continue;
            getRotatedRect(i);
            if(isCloseToBorder())
                continue;
            if(!isBlueNearby(i))
                continue;
            pushLights();
#if DRAW == SHOW_ALL
            drawLights();
#endif
        }
}


bool Armor::isAreaTooBigOrSmall(int i)
{
    int area = contourArea(V_contours[i]);
    return (area > AREA_MAX || area < AREA_MIN);
}

void Armor::getRotatedRect(int i)
{
    rotated_rect = minAreaRect(V_contours[i]);
}

bool Armor::isCloseToBorder()
{
            double recx = rotated_rect.center.x;
            double recy = rotated_rect.center.y;
            return (recx < BORDER_THRESHOLD || recx > srcW - BORDER_THRESHOLD
                    || recy < BORDER_THRESHOLD
                    || recy > srcH - BORDER_THRESHOLD);

}

bool Armor::isBlueNearby(int i)
{
    int blue_pixel_cnt = 0;
    uchar pixel = 0;
    uchar pixel_min = H_BLUE_LOW_THRESHOLD_MAX;
    uchar pixel_max = H_BLUE_HIGH_THRESHOLD_MIN;
    uchar pixel_S_select = 0;
    for(int j=0; j < (int)V_contours[i].size(); ++j)
    {
        pixel_S_select = *(hsvSplit[S_INDEX].ptr<uchar>(V_contours[i][j].y) + V_contours[i][j].x);
        pixel = *(hsvSplit[H_INDEX].ptr<uchar>(V_contours[i][j].y) + V_contours[i][j].x);
        if(pixel_S_select > S_BLUE_THRESHOLD && pixel < H_BLUE_HIGH_THRESHOLD && pixel > H_BLUE_LOW_THRESHOLD)
        {
            //cout << "pixel S:" << (int)pixel_S_select << " blue:" << (int)pixel << endl;
            if(pixel > pixel_max)
                pixel_max = pixel;
            if(pixel < pixel_min)
                pixel_min = pixel;
            ++blue_pixel_cnt;
        }
    }
    if( blue_pixel_cnt > 5 )
    {
        pixel_max = pixel_max < H_BLUE_HIGH_THRESHOLD_MAX ? pixel_max:H_BLUE_HIGH_THRESHOLD_MAX;
        pixel_max = pixel_max > H_BLUE_HIGH_THRESHOLD_MIN ? pixel_max:H_BLUE_HIGH_THRESHOLD_MIN;
        pixel_min = pixel_min > H_BLUE_LOW_THRESHOLD_MIN ? pixel_min:H_BLUE_LOW_THRESHOLD_MIN;
        pixel_min = pixel_min < H_BLUE_LOW_THRESHOLD_MAX ? pixel_min:H_BLUE_LOW_THRESHOLD_MAX;
        if(pixel_min > H_BLUE_LOW_THRESHOLD + H_BLUE_CHANGE_THRESHOLD_HIGH)
            H_BLUE_LOW_THRESHOLD += H_BLUE_STEP;
        if(pixel_min < H_BLUE_LOW_THRESHOLD + H_BLUE_CHANGE_THRESHOLD_LOW)
            H_BLUE_LOW_THRESHOLD -= H_BLUE_STEP;
        if(pixel_max < H_BLUE_HIGH_THRESHOLD - H_BLUE_CHANGE_THRESHOLD_HIGH)
            H_BLUE_HIGH_THRESHOLD -= H_BLUE_STEP;
        if(pixel_max > H_BLUE_HIGH_THRESHOLD - H_BLUE_CHANGE_THRESHOLD_LOW)
            H_BLUE_HIGH_THRESHOLD += H_BLUE_STEP;
    }
    //cout << "blue cnt:" << blue_pixel_cnt << "/size" << V_contours[i].size() << endl;
    return float(blue_pixel_cnt)/V_contours[i].size()*100 > BLUE_PIXEL_RATIO_THRESHOLD;
}

void Armor::pushLights()
{
    lights.push_back(rotated_rect);
}

#if DRAW == SHOW_ALL
void Armor::drawLights()
{
    Point2f vertices[4];
    rotated_rect.points(vertices);
    for (int j = 0; j < 4; ++j)
        line(light_draw, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 2);

    cv::namedWindow("draw", 1); 
    cv::imshow("draw", light_draw);
    cv::waitKey(1);

}
#endif

void Armor::selectLights()
{
    if (lights.size() > 1)
    {
        for (int i = 0; i < (int)lights.size() - 1; i++)
        {
            for (int j = i + 1; j < (int)lights.size(); j++)
            {
                Point2f pi = lights.at(i).center;
                Point2f pj = lights.at(j).center;
                double midx = (pi.x + pj.x) / 2;
                double midy = (pi.y + pj.y) / 2;
                Size2f sizei = lights.at(i).size;
                Size2f sizej = lights.at(j).size;
                double ai = sizei.height > sizei.width ? sizei.height : sizei.width;
                //                double b=sizei.height<sizei.width?sizei.height:sizei.width;
                double distance = sqrt((pi.x - pj.x) * (pi.x - pj.x) + (pi.y - pj.y) * (pi.y - pj.y));
                //灯条距离合适
                if (distance < 1.5 * ai || distance > 7.5 * ai)
                    continue;
                //灯条中点连线与灯条夹角合适
                double angeli = lights.at(i).angle;
                double angelj = lights.at(j).angle;
                if (sizei.width < sizei.height)
                    angeli += 90;
                if (sizej.width < sizej.height)
                    angelj += 90;
                double doti = abs(cos(angeli * PI / 180) * (pi.x - pj.x) + sin(angeli * PI / 180) * (pi.y - pj.y)) / distance;
                double dotj = abs(cos(angelj * PI / 180) * (pi.x - pj.x) + sin(angelj * PI / 180) * (pi.y - pj.y)) / distance;
                if (doti > 1.5 || dotj > 1.5)
                    continue;
                if(!isCircleAround(midx, midy))
                    continue;

                armors.push_back(Point((int)midx, (int)midy));
            }
        }
    }
#if DRAW == SHOW_ALL
    //for (int i=0;i<(int)armors.size();i++)
    {
        circle(light_draw, target, 3, cv::Scalar(0,0,255), -1);
    }
    imshow("draw", light_draw);
    cv::createTrackbar("H_BLUE_LOW_THRESHOLD", "draw", &H_BLUE_LOW_THRESHOLD, U8_MAX);
    cv::createTrackbar("H_BLUE_HIGH_THRESHOLD", "draw", &H_BLUE_HIGH_THRESHOLD, U8_MAX);
    cv::createTrackbar("S_BLUE_THRESHOLD", "draw", &S_BLUE_THRESHOLD, U8_MAX);
    cv::createTrackbar("BLUE_PIXEL_RATIO_THRESHOLD", "draw", &BLUE_PIXEL_RATIO_THRESHOLD, 100);

    cv::waitKey(1);
#endif
}

void Armor::chooseCloseTarget()
{
    if(!armors.empty())
    {
        int closest_x = 0, closest_y = 0;
        int distance_armor_center = 0;
        int distance_last = sqrt(
                (closest_x - srcW/2) * (closest_x - srcW/2)
                + (closest_y - srcH/2) * (closest_y - srcH/2));
        for(int i=0; i<(int)armors.size(); ++i)
        {
            distance_armor_center = sqrt(
                    (armors[i].x - srcW/2) * (armors[i].x - srcW/2)
                    + (armors[i].y - srcH/2) * (armors[i].y - srcH/2));
            if(distance_armor_center < distance_last)
            {
                closest_x = armors[i].x;
                closest_y = armors[i].y;
                distance_last = distance_armor_center;
            }
        }
        target.x = closest_x;
        target.y = closest_y;
    }
}

bool Armor::isCircleAround(int midx, int midy)
{
    Mat roi_circle = gray(
            cv::Rect(midx - CIRCLE_ROI_WIDTH/2, midy - CIRCLE_ROI_HEIGHT/2,
                CIRCLE_ROI_WIDTH, CIRCLE_ROI_HEIGHT));
    vector<vector<Point> > gray_contours;
    cv::threshold(roi_circle, roi_circle, CIRCLE_THRESHOLD, U8_MAX, THRESH_BINARY);
    //imshow("roi", roi_circle);
    cv::findContours(roi_circle, gray_contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    for(int i= 0; i < (int)gray_contours.size(); i++)
    {
        int area= contourArea(gray_contours.at(i));
        if(area > 50)
        {
            Point2f center(320, 240);
            float radius= 0;
            cv::minEnclosingCircle(gray_contours[i], center, radius);
            center.x += midx - CIRCLE_ROI_WIDTH/2;
            center.y += midy - CIRCLE_ROI_HEIGHT/2;
            int area= contourArea(gray_contours[i], false);
            float circleArea= PI * radius * radius;
            float r= area / circleArea;
            if(r > 0.7)
            {
#if DRAW == SHOW_ALL
                cv::circle(light_draw, center, radius, Scalar(0, 255, 255), 2);
#endif
                return true;
            }
        }
    }
    return false;
}

void Armor::cleanAll()
{
    V_contours.clear();
    lights.clear();
    armors.clear();
}

bool Armor::isFound()
{
    return !armors.empty();
}

int Armor::getTargetX()
{
    return target.x;
}

int Armor::getTargetY()
{
    return target.y;
}

int main(void)
{
    //cv::VideoCapture cap(1);
    cv::VideoCapture cap("/home/jachinshen/视频/Robo/BlueArmor2.avi"); 
    Mat src;
    Armor armor;
    cv::namedWindow("frame", 1); 
    cap.read(src);
    imshow("frame", src);
    waitKey(0);

    armor.init();
    while(cap.read(src))	
    {
        armor.feedImage(src);
        cout << "x:" << armor.getTargetX()
            << " y:" << armor.getTargetY() << endl;

        imshow("frame", src);
        waitKey(1);
    }
    cap.release();
}
