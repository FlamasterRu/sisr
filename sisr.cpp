#include "sisr.h"

uchar hashUnity = 1;


cv::Mat VaveletHaara(const cv::Mat& image)  // Возвращает аппроксимированную часть изображение из дискретного вейвлет преобразования Хаара
{
    if (image.rows != image.cols)
        throw("image.rows != image.cols");

    cv::Mat res1(image.rows, image.cols%2==0 ? image.cols/2 : image.cols/2+1, image.type());
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols/2; ++j)
        {
            int v1 = image.at<uchar>(i, 2*j);
            int v2 = image.at<uchar>(i, 2*j+1);
            res1.at<uchar>(i, j) = static_cast<uchar>( (v1 + v2) / 2);
            //std::cout << i << " " << j << " " << v1 << " " << v2 << " " << (int)res1.at<uchar>(i, j) << std::endl;
        }
        if (image.cols%2 != 0)
        {
            res1.at<uchar>(i, res1.cols-1) = image.at<uchar>(i, image.cols-1);
        }
    }

    cv::Mat res2(image.rows%2==0 ? image.rows/2 : image.rows/2+1, res1.cols, image.type());
    //std::cout << res1 << std::endl;
    for (int i = 0; i < res1.rows/2; ++i)
    {
        for (int j = 0; j < res1.cols; ++j)
        {
            int v1 = res1.at<uchar>(2*i, j);
            int v2 = res1.at<uchar>(2*i+1, j);
            res2.at<uchar>(i, j) = static_cast<uchar>( (v1 + v2) / 2);
            //std::cout << i << " " << j << " " << v1 << " " << v2 << " " << (int)res2.at<uchar>(i, j) << std::endl;
        }
    }
    if (image.rows%2 != 0)
    {
        for (int j = 0; j < res1.cols; ++j)
        {
            res2.at<uchar>(res2.rows-1, j) = res1.at<uchar>(res1.rows-1, j);
        }
    }
    return res2;
}

uint qHash(const cv::Mat& image)
{
    cv::Mat cop(image);
    //std::cout << cop << std::endl;
    while (cop.rows + cop.cols > 4)
    {
        cop = VaveletHaara(cop);
    }
    uchar data[4];
    data[0] = cop.at<uchar>(0, 0)/hashUnity;
    data[1] = cop.at<uchar>(0, 1)/hashUnity;
    data[2] = cop.at<uchar>(1, 0)/hashUnity;
    data[3] = cop.at<uchar>(1, 1)/hashUnity;
    uint res = 1;
    memcpy(&res, data, 4);
    return res;
}

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

    cv::Mat mat(3, 3, image.type());
    mat.at<uchar>(0, 0) = 1;
    mat.at<uchar>(0, 1) = 2;
    mat.at<uchar>(0, 2) = 3;
    mat.at<uchar>(1, 0) = 4;
    mat.at<uchar>(1, 1) = 5;
    mat.at<uchar>(1, 2) = 6;
    mat.at<uchar>(2, 0) = 7;
    mat.at<uchar>(2, 1) = 8;
    mat.at<uchar>(2, 2) = 9;

//    mat.at<uchar>(0, 0) = 1;
//    mat.at<uchar>(0, 1) = 2;
//    mat.at<uchar>(0, 2) = 3;
//    mat.at<uchar>(0, 3) = 4;
//    mat.at<uchar>(1, 0) = 5;
//    mat.at<uchar>(1, 1) = 6;
//    mat.at<uchar>(1, 2) = 7;
//    mat.at<uchar>(1, 3) = 8;
//    mat.at<uchar>(2, 0) = 9;
//    mat.at<uchar>(2, 1) = 10;
//    mat.at<uchar>(2, 2) = 11;
//    mat.at<uchar>(2, 3) = 12;
//    mat.at<uchar>(3, 0) = 13;
//    mat.at<uchar>(3, 1) = 14;
//    mat.at<uchar>(3, 2) = 15;
//    mat.at<uchar>(3, 3) = 16;

//    mat.at<uchar>(0, 0) = 1;
//    mat.at<uchar>(0, 1) = 2;
//    mat.at<uchar>(0, 2) = 3;
//    mat.at<uchar>(0, 3) = 4;
//    mat.at<uchar>(0, 4) = 5;
//    mat.at<uchar>(1, 0) = 6;
//    mat.at<uchar>(1, 1) = 7;
//    mat.at<uchar>(1, 2) = 8;
//    mat.at<uchar>(1, 3) = 9;
//    mat.at<uchar>(1, 4) = 10;
//    mat.at<uchar>(2, 0) = 11;
//    mat.at<uchar>(2, 1) = 12;
//    mat.at<uchar>(2, 2) = 13;
//    mat.at<uchar>(2, 3) = 14;
//    mat.at<uchar>(2, 4) = 15;
//    mat.at<uchar>(3, 0) = 16;
//    mat.at<uchar>(3, 1) = 17;
//    mat.at<uchar>(3, 2) = 18;
//    mat.at<uchar>(3, 3) = 19;
//    mat.at<uchar>(3, 4) = 20;
//    mat.at<uchar>(4, 0) = 21;
//    mat.at<uchar>(4, 1) = 22;
//    mat.at<uchar>(4, 2) = 23;
//    mat.at<uchar>(4, 3) = 24;
//    mat.at<uchar>(4, 4) = 25;



    //std::cout << mat << std::endl;
    //cv::Mat tmp = VaveletHaara(mat);
    //std::cout << tmp << std::endl;
    //std::cout << qHash(mat) << std::endl;

    return true;
}

bool SISR::CreateLRHRPairs()
{
    cv::Mat curImage;
    mLRImage.copyTo(curImage);

    uint maxSize = ((uint)curImage.rows - 8u)*((uint)curImage.cols - 8u);
    std::cout << "Hash teor size = " << maxSize << std::endl;
    if (maxSize < 16u)
        hashUnity = 128;
    else if (maxSize < 256u)
        hashUnity = 64;
    else if (maxSize < 4096u)
        hashUnity = 32;
    else if (maxSize < 65536u)
        hashUnity = 16;
    else if (maxSize < 1048576u)
        hashUnity = 8;
    else if (maxSize < 16777216u)
        hashUnity = 4;
    else if (maxSize < 268435456u)
        hashUnity = 2;
    else
        hashUnity = 1;
    std::cout << "Hash unity = " << (uint)hashUnity << std::endl;

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
                mHash[qHash(LR)].push_back(mPairs.size()-1);    // хэш по LR патчу хранит номер пары в списке пар
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
    sumT1 = 0;
    sumT2 = 0;
    for (int i = 0; i < mLRImage.rows - 2; ++i)
    {
        //std::cout << i << "/" << mLRImage.rows - 2 << std::endl;
        for (int j = 0; j < mLRImage.cols - 2; ++j)
        {
            cv::Mat LRpart(mLRImage, cv::Rect(i, j, 3, 3));
            //GetNearestPairsIDS(LRpart, nearestPairs, dist);
            GetNearestPairsIDS2(LRpart, nearestPairs, dist);
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
    std::cout << "GetNearestPairsIDS " << sumT1/1000. << std::endl;
    std::cout << "AssemblyHRPatch " << sumT2/1000. << std::endl;
    return true;
}


void SISR::GetNearestPairsIDS(const cv::Mat& part, QList<int>& nearest, QList<double>& dist)
{
    //QTime t;
    //t.restart();
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
    //sumT1 += t.restart();
}

void SISR::GetNearestPairsIDS2(const cv::Mat& part, QList<int>& nearest, QList<double>& dist)
{
    QTime t;
    t.restart();
    if (mHash.contains(qHash(part)))
    {
        nearest.clear();
        dist.clear();
        QList<QPair<int, double>> tmp;
        //std::cout << qHash(part) << " " << mHash[qHash(part)].size() << std::endl;
        for (uint idx : mHash[qHash(part)])
        {
            double d = StandartDerivation(part, mPairs[idx].first);
            tmp.push_back(QPair<int,double>(idx, d));
        }
        if (tmp.size() >= 4)
        {
            std::partial_sort(tmp.begin(), tmp.begin() + 4, tmp.end(), [](const QPair<int,double>& l, const QPair<int,double>& r)
            { return l.second < r.second; });
            for (int i = 0; i < 4; ++i)
            {
                nearest.push_back(tmp[i].first);
                dist.push_back(tmp[i].second);
            }
        }
        else
        {
            for (int i = 0; i < tmp.size(); ++i)
            {
                nearest.push_back(tmp[i].first);
                dist.push_back(tmp[i].second);
            }
        }
    }
    else
    {
        //std::cout << "NOPE " << qHash(part) << std::endl;;
        GetNearestPairsIDS(part, nearest, dist);
    }
    sumT1 += t.restart();
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
    QTime t;
    t.restart();
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
    sumT2 += t.restart();
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
