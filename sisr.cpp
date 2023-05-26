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
    cv::resize(mHRImage, mHRImage, cv::Size(mLRImage.cols, mLRImage.rows), cv::INTER_CUBIC);
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
    std::partial_sort(tmp.begin(), tmp.begin() + 8, tmp.end(), [](const QPair<int,double>& l, const QPair<int,double>& r)
    { return l.second < r.second; });
    for (int i = 0; i < 8; ++i)
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
    QList<double> dist; // расстояние по модулю между пикселями
    for (int i = 0; i < i1.rows; ++i)
    {
        for (int j = 0; j < i2.cols; ++j)
        {
            dist.push_back(std::abs(i1.at<uchar>(i,j) - i2.at<uchar>(i,j)));
        }
    }
    double aver = 0.;   // среднее арифметическое расстояний
    for (int d : dist)
    {
        aver += d;
    }
    aver /= static_cast<double>(dist.size());

    for (double& d : dist)  // квадраты отклонений расстояния от среднего
    {
        d = (d-aver)*(d-aver);
    }
    aver = 0.;
    for (int d : dist)  // дисперсия
    {
        aver += d;
    }
    aver /= static_cast<double>(dist.size()-1);
    return sqrt(aver);   // среднеквадратичное отклонение
}

cv::Mat SISR::AssemblyHRPatch(QList<int> nearest, QList<double> weight)
{
    cv::Mat result;
    mPairs[nearest.first()].second.copyTo(result);

    double sum = 0.;
    for (double w : weight)
    {
        sum += w;
    }

    for (int i = 0; i < result.rows; ++i)
    {
        for (int j = 0; j < result.cols; ++j)
        {
            double p = 0.;
            for (int k = 0; k < nearest.size(); ++k)
            {
                double w = weight[k] / sum;
                p += w * static_cast<double>( mPairs[nearest[k]].second.at<uchar>(i, j) );
            }
            result.at<uchar>(i,j) = (uchar)p;
        }
    }
    return result;
}




