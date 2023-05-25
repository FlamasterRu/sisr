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
        cv::resize(curImage, curImage, cv::Size(curImage.rows*0.95, curImage.cols*0.95), cv::INTER_NEAREST);
    }

    return true;
}

bool SISR::AssemblyHRImage()
{
    QList<int> nearestPairs;
    QList<double> dist;
    for (int i = 0; i < mLRImage.rows - 3; ++i)
    {
        for (int j = 0; j < mLRImage.cols - 3; ++j)
        {
            cv::Mat LRpart(mLRImage, cv::Rect(i, j, 3, 3));
            GetNearestPairsIDS(LRpart, nearestPairs, dist);
            if (nearestPairs.size() != 0)
            {
                Patch p;
                p.rect = cv::Rect(i, j, 3, 3);
                p.patchNums = nearestPairs;
                p.distToPatches = dist;
                p.LR = LRpart;
                p.HR = AssemblyHRPatch(nearestPairs, dist);
                mPatches.push_back(p);
            }
            else
            {
                std::cout << i << " " << j << std::endl;
            }
        }
    }

    return true;
}

void SISR::GetNearestPairsIDS(const cv::Mat& part, QList<int>& nearest, QList<double>& dist)
{
    nearest.clear();
    dist.clear();
    for (int i = 0; i < mPairs.size(); ++i)
    {
        double d = EuclidDist(part, mPairs[i].first);
        if (d < 10.)
        {
            nearest.push_back(i);
            dist.push_back(d);
        }
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




