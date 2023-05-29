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

int SISR::GetPatchesCount()
{
    return mPatches.size();
}

Patch SISR::GetPatch(int i)
{
    return mPatches.at(i);
}

cv::Mat SISR::GetHRImage()
{
    return mHRImage;
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
    while (curImage.rows > 9 && curImage.cols > 9)  // ищем во всех разрешениях, с уменьшением в 0.9 раз
    {
        for (int i = 0; i < curImage.rows - 8; ++i)
        {
            for (int j = 0; j < curImage.cols - 8; ++j)
            {
                cv::Mat HR(curImage, cv::Rect(i, j, 9, 9));
                cv::Mat LR;
                cv::resize(HR, LR, cv::Size(3, 3), cv::INTER_NEAREST);
                mPairs.push_back(MPair(LR, HR));
            }
        }
        cv::resize(curImage, curImage, cv::Size(curImage.rows*0.95, curImage.cols*0.95), cv::INTER_NEAREST);
    }

    return true;
}

bool SISR::AssemblyHRImage()
{
    QList<int> nearestPairs;
    QList<double> dist;
    mHRImage.create(mLRImage.rows*2 + 3, mLRImage.cols*2 + 3, CV_8UC1);
    for (int i = 0; i < mHRImage.rows; ++i)
    {
        for (int j = 0; j < mHRImage.cols; ++j)
        {
            mHRImage.at<uchar>(i,j) = 240;
        }
    }
    for (int i = 0; i < mLRImage.rows - 2; ++i)
    {
        std::cout << i << "/" << mLRImage.rows - 2 << std::endl;
        for (int j = 0; j < mLRImage.cols - 2; ++j)
        {
            cv::Mat LRpart(mLRImage, cv::Rect(i, j, 3, 3));
            GetNearestPairsIDS(LRpart, nearestPairs, dist);
            Patch p;
            p.rect = cv::Rect(i, j, 3, 3);
            p.patchNums = nearestPairs;
            p.distToPatches = dist;
            p.LR = LRpart;
            p.HR = AssemblyHRPatch(nearestPairs, dist);
            mPatches.push_back(p);
            p.HR.copyTo(mHRImage(cv::Rect((i+1)*2-2, (j+1)*2-2, p.HR.cols, p.HR.rows)));
        }
    }
    cv::resize(mHRImage, mHRImage, cv::Size(mLRImage.cols*2, mLRImage.rows*2), cv::INTER_CUBIC);
    return true;
}


void SISR::GetNearestPairsIDS(const cv::Mat& part, QList<int>& nearest, QList<double>& dist)
{
    nearest.clear();
    dist.clear();
    QList<QPair<int, double>> tmp;
    for (int i = 0; i < mPairs.size(); ++i)
    {
        double d = StandartDerivation(part, mPairs[i].first);
        tmp.push_back(QPair<int,double>(i, d));
    }
    std::partial_sort(tmp.begin(), tmp.begin() + 4, tmp.end(), [](const QPair<int,double>& l, const QPair<int,double>& r)
    { return l.second < r.second; });
    for (int i = 0; i < 4; ++i)
    {
        nearest.push_back(tmp[i].first);
        dist.push_back(tmp[i].second);
    }
}

double SISR::EuclidDist(const cv::Mat& i1, const cv::Mat& i2)
{
    double res = 0.;
    for (int i = 0; i < i1.rows; ++i)
    {
        for (int j = 0; j < i2.cols; ++j)
        {
            res += (i1.at<uchar>(i,j) - i2.at<uchar>(i,j))*(i1.at<uchar>(i,j) - i2.at<uchar>(i,j));
        }
    }
    return std::sqrt(res);
}

double SISR::StandartDerivation(const cv::Mat& i1, const cv::Mat& i2)
{
    double res = 0.;
    for (int i = 0; i < i1.rows; ++i)
    {
        for (int j = 0; j < i2.cols; ++j)
        {
            double d = static_cast<double>(i1.at<uchar>(i,j)) - static_cast<double>(i2.at<uchar>(i,j));
            res += d*d;
        }
    }
    return res/(i1.rows*i2.cols);
}

cv::Mat SISR::AssemblyHRPatch(QList<int> nearest, QList<double> weight)
{
    cv::Mat result;
    mPairs[nearest.first()].second.copyTo(result);

    for (int i = 0; i < result.rows; ++i)
    {
        for (int j = 0; j < result.cols; ++j)
        {
            double p = 0.;
            for (int k = 0; k < nearest.size(); ++k)
            {
                p += static_cast<double>( mPairs[nearest[k]].second.at<uchar>(i, j) );
            }
            p /= static_cast<double>(nearest.size());
            result.at<uchar>(i,j) = (uchar)p;
        }
    }
    return result;
}




double SISR::RMSE(const cv::Mat& image1, const cv::Mat& image2)
{
    double res = 0.;
    for (int i = 0; i < image1.rows; ++i)
    {
        for (int j = 0; j < image1.cols; ++j)
        {
            double v1 = image1.at<uchar>(i,j);
            double v2 = image2.at<uchar>(i,j);
            res += (v1-v2)*(v1-v2);
        }
    }
    res /= static_cast<double>(image1.rows*image1.cols);
    return sqrt(res);
}

double SISR::MaxDeviation(const cv::Mat& image1, const cv::Mat& image2)
{
    double res = 0.;
    for (int i = 0; i < image1.rows; ++i)
    {
        for (int j = 0; j < image1.cols; ++j)
        {
            double v1 = image1.at<uchar>(i,j);
            double v2 = image2.at<uchar>(i,j);
            double d = abs(v1 - v2);
            if (d > res)
            {
                res = d;
            }
        }
    }
    return res;
}

double SISR::PSNR(const cv::Mat& image1, const cv::Mat& image2)
{
    double rmse = RMSE(image1, image2);
    double max = Max(image1);
    return 20.*std::log10(max/rmse);
}

double SISR::SSIM(const cv::Mat& image1, const cv::Mat& image2)
{
    double ave1 = 0, ave2 = 0;
    for (int i = 0; i < image1.rows; ++i)
    {
        for (int j = 0; j < image1.cols; ++j)
        {
            ave1 += static_cast<double>(image1.at<uchar>(i,j));
            ave2 += static_cast<double>(image2.at<uchar>(i,j));
        }
    }
    ave1 /= static_cast<double>(image1.rows*image1.cols);
    ave2 /= static_cast<double>(image2.rows*image2.cols);

    double sco1 = 0, sco2 = 0;
    for (int i = 0; i < image1.rows; ++i)
    {
        for (int j = 0; j < image1.cols; ++j)
        {
            double v1 = ave1 - static_cast<double>(image1.at<uchar>(i,j));
            double v2 = ave2 - static_cast<double>(image2.at<uchar>(i,j));
            sco1 += (ave1 - v1)*(ave1 - v1);
            sco2 += (ave2 - v2)*(ave2 - v2);
        }
    }
    sco1 /= static_cast<double>(image1.rows*image1.cols);
    sco2 /= static_cast<double>(image2.rows*image2.cols);

    double cov = 0;
    for (int i = 0; i < image1.rows; ++i)
    {
        for (int j = 0; j < image1.cols; ++j)
        {
            double v1 = static_cast<double>(image1.at<uchar>(i,j));
            double v2 = static_cast<double>(image2.at<uchar>(i,j));
            cov += (v1 - ave1)*(v2 - ave2);
        }
    }
    cov /= static_cast<double>(image1.rows*image1.cols);

    double c1 = 0.01*255*0.01*255;
    double c2 = 0.03*0.03*255*255;
    return (2.*ave1*ave2 + c1)*(2.*cov + c2) /
            ( (ave1*ave1 + ave2*ave2 + c1)*(sco1*sco1 + sco2*sco2 + c2) );
}

double SISR::Max(const cv::Mat& image)
{
    double res = 0.;
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            double v = image.at<uchar>(i,j);
            if (v > res)
            {
                res = v;
            }
        }
    }
    return res;
}
