#include "sisr.h"

SISR::SISR()
{

}

int SISR::GetPairsCount()
{
    return mPairs.size();
}

MPair SISR::GetPair(int i)
{
    return mPairs.at(i);
}


bool SISR::InitImage(const cv::Mat& image)
{
    image.copyTo(mLRImage);
    return true;
}

bool SISR::CreateLRHRPairs()
{
    cv::Mat curImage;
    mLRImage.copyTo(curImage);
    while (curImage.rows > 10 && curImage.cols > 10)  // ищем во всех разрешениях, с уменьшением в 0.9 раз
    {
        for (int i = 0; i < curImage.rows - 9; ++i)
        {
            for (int j = 0; j < curImage.cols - 9; ++j)
            {
                cv::Mat HR(curImage, cv::Rect(i, j, 9, 9));
                cv::Mat LR;
                cv::resize(HR, LR, cv::Size(3, 3), cv::INTER_NEAREST);
                mPairs.push_back(MPair(LR, HR));
            }
        }
        cv::resize(curImage, curImage, cv::Size(curImage.rows*0.9, curImage.cols*0.9), cv::INTER_NEAREST);
    }

    return true;
}








