#define MANIFOLD 1
#define PC 0
//#if defined (__amd64__) || ( __amd64) ||(__x86_64__) || (__x86_64) ||(i386) ||(__i386) ||(__i386__)
#if defined __arm__
#define PLATFORM MANIFOLD
#else
#define PLATFORM PC
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//using namespace cv;

#if PLATFORM == PC
typedef cv::Mat TMat;
#else
typedef cv::gpu::GPUMat TMat;
#endif

#include <iostream>
#include <vector>
using namespace std;

#include <stdlib.h>
#include <sys/time.h>

#define SHOW_ALL SHOW_GRAY|SHOW_LIGHT_REGION|SHOW_DRAW|SHOW_ROI
#define SHOW_ROI  0x08
#define SHOW_DRAW 0x04
#define SHOW_GRAY 0x02
#define SHOW_LIGHT_REGION 0x01
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
        int CIRCLE_ROI_WIDTH;
        int CIRCLE_ROI_HEIGHT;
        int CIRCLE_THRESHOLD;
        int CIRCLE_AREA_THRESH_MAX;
        int CIRCLE_AREA_THRESH_MIN;
        int DRAW;
        bool is_last_found;
        int refresh_ctr;
        //double fps;

    private:
        //cv::Mat hsv;
        //cv::Mat s_low;
        //cv::Mat s_canny;
        //cv::Mat v_very_high;
        //cv::Mat gray;
        cv::Mat light_draw;

        //vector<cv::Mat > hsvSplit;
        vector<cv::RotatedRect> lights;
        vector<cv::Point2f > armors;
        vector<vector<cv::Point > > V_contours;

        cv::Mat V_element_erode;
        cv::Mat V_element_dilate;

        int srcH, srcW;
        cv::Point target;

    private:
        void cvtHSV(const cv::Mat& src, vector<TMat >& hsvSplit);
        void cvtGray(const cv::Mat& src, TMat& gray);
        void getLightRegion(vector<TMat >& hsvSplit, TMat& v_very_high);
        void selectContours(vector<TMat >& hsvSplit);
        bool isBlueNearby(vector<TMat >& hsvSplit, vector<cv::Point>& contour);
        void selectLights(const cv::Mat& src);
        bool isCircleAround(cv::Mat& gray, int midx, int midy);
        void findCircleAround(const cv::Mat& src);
        bool isCloseToBorder(cv::RotatedRect& rotated_rect);
        bool isAreaTooBigOrSmall(vector<cv::Point>& contour);
        void getSrcSize(const cv::Mat& src);
        void chooseCloseTarget();
        void cleanAll();
        double tic();

    public:
        Armor();
        void init(const cv::Mat& src);
        void feedImage(const cv::Mat& src);
        bool isFound();
        int getTargetX();
        int getTargetY();
        void setDraw(int is_draw);
};

