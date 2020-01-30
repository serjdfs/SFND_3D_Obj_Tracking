
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        {
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0;
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    cv::KeyPoint left, right;
    double sum = 0.0;
    for(auto &kptMatch : kptMatches){
        sum += kptMatch.distance;
    }
    sum /= kptMatches.size();

    for(auto &kptMatch : kptMatches){
        left = kptsPrev[kptMatch.trainIdx];
        right = kptsCurr[kptMatch.queryIdx];
        if(boundingBox.roi.contains(left.pt) && boundingBox.roi.contains(right.pt)){
            if(sum <= kptMatch.distance)
                boundingBox.kptMatches.push_back(kptMatch);
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = 0;

    if (distRatios.size() % 2 != 0)
        medianDistRatio = (double)distRatios[distRatios.size() / 2.0];
    else
        medianDistRatio = (distRatios[(distRatios.size() - 1) / 2.0] + distRatios[distRatios.size() / 2.0]) / 2.0;

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
    cout << "TTC Camera: " << TTC << endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary variables
    double dT = 1 / frameRate; // time between two measurements in seconds

    // find closest distance to Lidar points
    double minXPrev = 1e9, minXCurr = 1e9;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) {
        minXPrev = minXPrev>it->x ? it->x : minXPrev;
    }

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) {
        minXCurr = minXCurr>it->x ? it->x : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev-minXCurr);
    cout << "TTC Lidar: " << TTC << endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches,
                        DataFrame &prev_frame, DataFrame &curr_frame) {

    int match_counts[curr_frame.boundingBoxes.size()][prev_frame.boundingBoxes.size()];
    for (int i = 0; i < curr_frame.boundingBoxes.size(); i++)
        for (int j = 0; j < prev_frame.boundingBoxes.size(); j++)
            match_counts[i][j] = 0;

    for(auto &match:matches){
        for(auto &prev : prev_frame.boundingBoxes){
            for(auto &curr : curr_frame.boundingBoxes){
                auto query_pt = prev_frame.keypoints[match.queryIdx].pt;
                auto train_pt = curr_frame.keypoints[match.trainIdx].pt;
                if(prev.roi.contains(query_pt) && curr.roi.contains(train_pt)){
                    ++match_counts[curr.boxID][prev.boxID];
                }
            }
        }
    }

    static const int MIN_BBOX_MATCH_THSLD = 1;

    bool processed[prev_frame.boundingBoxes.size()];
    for(int i = 0; i < curr_frame.boundingBoxes.size(); ++i){
        int max = 0, argmax = 0;
        for(int j = 0; j < prev_frame.boundingBoxes.size(); ++j){
            if(match_counts[i][j]>max){
                max = match_counts[i][j];
                argmax = j;
            }
        }
        if(max > MIN_BBOX_MATCH_THSLD && !processed[argmax]){
            bbBestMatches.insert({argmax, i});
            processed[argmax] = true;
        }
    }

    /* Now visualize bounding box matches
    // copy the previous frame and current frame
    cv::Mat prv_img = prev_frame.cameraImg.clone();
    cv::Mat cur_img = curr_frame.cameraImg.clone();

    const auto red = cv::Scalar(0, 0, 255);
    const auto green = cv::Scalar(0, 255, 0);
    const auto blue = cv::Scalar(255, 0, 0);

    cv::putText(prv_img, "Previous Frame", cv::Point(50, 50), cv::FONT_ITALIC, 0.75, red, 2);
    cv::putText(cur_img, "Current Frame", cv::Point(50, 50), cv::FONT_ITALIC, 0.75, blue, 2);

    std::vector<std::pair<cv::Point2i,cv::Point2i>> lines;

    for (auto const &kv : bbBestMatches) {
        std::cout << "BBOX MATCH IDS: " << std::to_string(kv.first) + " -> " + std::to_string(kv.second) << std::endl;

        auto prv_rect = prev_frame.boundingBoxes[kv.first].roi;
        cv::rectangle(prv_img, cv::Point(prv_rect.x, prv_rect.y), cv::Point(prv_rect.x + prv_rect.width, prv_rect.y + prv_rect.height), red, 1);

        auto cur_rect = curr_frame.boundingBoxes[kv.second].roi;
        cv::rectangle(cur_img,cv::Point(cur_rect.x, cur_rect.y),cv::Point(cur_rect.x + cur_rect.width, cur_rect.y + cur_rect.height),blue,1);

        lines.emplace_back(cv::Point(prv_rect.x, prv_rect.y), cv::Point(cur_rect.x, cur_rect.y+prv_img.size().height) );
    }

    cv::Mat concat_img;
    cv::vconcat(prv_img, cur_img, concat_img);

    for(auto &end_pts : lines ){
        cv::line(concat_img, end_pts.first, end_pts.second, green, 1);
    }

    string windowName = "Matching Bounding Boxes";
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, concat_img);
    cv::waitKey(0); // wait for key to be pressed
    */
}
