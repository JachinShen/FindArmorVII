#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <iostream>
#include <vector>
using namespace std;

#define DRAW SHOW_ALL
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

    private:
        cv::Mat hsv;
        cv::Mat s_low;
        cv::Mat s_canny;
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
        void getLightRegion();
        void getLightRegion2();
        void selectContours();
        bool isAreaTooBigOrSmall(int i);
        void getRotatedRect(int i);
        bool isCloseToBorder();
        bool isNoBlueNearby(int i);
        void pushLights();
        void drawLights();
        void selectLights();
        void cleanAll();

    public:
        Armor();
        void init();
        void feedImage(cv::Mat& src);
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
    H_BLUE_STEP(5),
    H_BLUE_CHANGE_THRESHOLD_LOW(5),
    H_BLUE_CHANGE_THRESHOLD_HIGH(10),
    S_BLUE_THRESHOLD(100),
    BLUE_PIXEL_RATIO_THRESHOLD(12),
    R_MAX_THRESHOLD(15),
    R_MIN_THRESHOLD(5),
    FLOOD_LOWER(8),
    FLOOD_UPPER(8)
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
    getLightRegion();
    findContours(hsvSplit[V_INDEX], V_contours,
            CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    selectContours(); 
    selectLights();
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

void Armor::getLightRegion()
{
    cv::threshold(hsvSplit[V_INDEX], hsvSplit[V_INDEX],
            V_THRESHOLD, U8_MAX, THRESH_BINARY);
    cv::threshold(hsvSplit[S_INDEX], s_low,
            S_THRESHOLD, U8_MAX, THRESH_BINARY_INV);
    bitwise_and(s_low, hsvSplit[V_INDEX], hsvSplit[V_INDEX]);
    cv::erode(hsvSplit[V_INDEX], hsvSplit[V_INDEX], V_element_erode);
    cv::dilate(hsvSplit[V_INDEX], hsvSplit[V_INDEX], V_element_dilate);
#if DRAW == SHOW_ALL
    cv::namedWindow("light region",1);
    cv::imshow("light region", hsvSplit[V_INDEX]);
    cv::imshow("S region", hsvSplit[S_INDEX]);
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
            if(isNoBlueNearby(i))
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

bool Armor::isNoBlueNearby(int i)
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
    return float(blue_pixel_cnt)/V_contours[i].size()*100 < BLUE_PIXEL_RATIO_THRESHOLD;
}

void Armor::pushLights()
{
    lights.push_back(rotated_rect);
}

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

void Armor::selectLights()
{
    /*
    uchar *data = NULL;
    float r_real = 0;
    unsigned int circle_point_cnt = 0;
    */
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
                if (doti > 0.9 || dotj > 0.9)
                    continue;
                /*
                cv::Canny(hsvSplit[S_INDEX], s_canny, 300, 600);
                imshow("circle", s_canny);
                cv::createTrackbar("R_MAX_THRESHOLD", "circle", &R_MAX_THRESHOLD, U8_MAX);
                cv::createTrackbar("R_MIN_THRESHOLD", "circle", &R_MIN_THRESHOLD, U8_MAX);
                for(int row=0; row<srcH; ++row)
                {
                    data = s_canny.ptr<uchar>(row);
                    for(int col=0; col<srcW; ++col, ++data)
                    {
                        if(*data)
                        {
                            r_real = sqrt((midx - col) * (midx - col) + (midy - row) * (midy - row));
                            if(R_MIN_THRESHOLD< r_real && r_real < R_MAX_THRESHOLD)
                            {
                                cout << "R: " << r_real << endl;
                                ++circle_point_cnt;
                            }
                        }
                    }
                }
                cout << "cnt: " << circle_point_cnt << endl;
                cv::Rect rect_center;
                cv::floodFill(hsvSplit[S_INDEX], Point((int)midx, (int)midy),
                        cv::Scalar(255), &rect_center, cv::Scalar(FLOOD_LOWER), cv::Scalar(FLOOD_UPPER));
                cv::rectangle(light_draw, rect_center, cv::Scalar(255, 0, 0), 2);
                cout << "Rect x: " << rect_center.x << " y: " << rect_center.y << endl;
                */
                armors.push_back(Point((int)midx, (int)midy));
            }
        }
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

void Armor::cleanAll()
{
    V_contours.clear();
    lights.clear();
    armors.clear();
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

        imshow("frame", src);
        waitKey(0);
    }
    cap.release();
}
        /*
        GaussianBlur(hsvSplit[2], hsvSplit[2], Size(9, 9), 0);
        minMaxLoc(hsvSplit[2], NULL, &V_max_value, NULL, &V_max_loc); 
        threshold(hsvSplit[2], hsvSplit[2], V_thresh_ratio * V_max_value, 255, THRESH_BINARY);
        erode(hsvSplit[2], hsvSplit[2], V_element_erode);
        dilate(hsvSplit[2], hsvSplit[2], V_element_dilate);
        imshow("V", hsvSplit[2]);
        //findContours(hsvSplit[2], V_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        //cout << countNonZero(hsvSplit[2]) << endl;
        //cout << V_contours.size() << endl;

        
        threshold(hsvSplit[1], hsvSplit[1], 30, 255, THRESH_BINARY_INV);
        imshow("S", hsvSplit[1]);
        
        bitwise_and(hsvSplit[1], hsvSplit[2], light_blue);
        imshow("light_blue", light_blue);
        findContours(light_blue, S_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        cout << S_contours.size() << endl;
        */

        /*
        GaussianBlur(hsvSplit[2], hsvSplit[2], Size(7, 7), 0);
        threshold(hsvSplit[2], hsvSplit[2], 230, 255, THRESH_BINARY);
        imshow("V", hsvSplit[2]);

        findContours(hsvSplit[2], S_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        for(int i=0; i < (int)S_contours.size(); ++i)
        {
            bounding_rect = boundingRect(S_contours[i]);
            Mat roi = src(bounding_rect);
            Scalar bgr_mean = mean(roi);
            cout<< i<<" "<<bgr_mean<<endl;
        }

        GaussianBlur(hsvSplit[0], hsvSplit[0], Size(9, 9), 0);
        threshold(hsvSplit[0], h_low, H_LOW_THRESHOLD, 255, THRESH_BINARY);
        threshold(hsvSplit[0], h_high, H_HIGH_THRESHOLD, 255, THRESH_BINARY_INV);
        bitwise_and(h_low, h_high, hsvSplit[0]);
        imshow("H", hsvSplit[0]);
        //imshow("light_blue", light_blue);
        */
        //minMaxLoc(hsvSplit[1], &S_min_value, &S_max_value, NULL, NULL); 
        //adaptiveThreshold(hsvSplit[1], hsvSplit[1], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, -60);
        /*
        try
        {
            //Canny(hsvSplit[1], hsvSplit[1], canny_threshold_1*100.0, canny_threshold_2, 7);
            adaptiveThreshold(hsvSplit[1], hsvSplit[1], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, -adaptive_C);
            imshow("S", hsvSplit[1]);
            //createTrackbar("canny_threshold_1", "S", &canny_threshold_1, 30000);
            //createTrackbar("canny_threshold_2", "S", &canny_threshold_2, 500);
            createTrackbar("adaptive_C", "S", &adaptive_C, 500);
            findContours(hsvSplit[1], S_contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            drawContours(src, S_contours, -1, Scalar(0,255,0));

            imshow("frame", src);
            waitKey(0);
        }
        catch( cv::Exception& e )
        {
            cout << "Error: " << e.what() << endl;
        }
        */


