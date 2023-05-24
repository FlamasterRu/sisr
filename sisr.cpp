#include "sisr.h"

SISR::SISR()
{

}

bool SISR::InitImage(const QString& fullFileName)
{
    mFullFileName = fullFileName;
    mStartImage = cv::imread(fullFileName.toStdString());
    cv::cvtColor(mStartImage, mStartImage, cv::COLOR_BGR2RGB);
    cv::resize(mStartImage, mLRImage, cv::Size(mStartImage.cols/2, mStartImage.rows/2), cv::INTER_CUBIC);
    return true;
}

bool SISR::SplitColors()
{
    mLRImage.copyTo(mLRRed);
    mLRImage.copyTo(mLRGreen);
    mLRImage.copyTo(mLRBlue);

    cv::cvtColor(mLRImage, mLRGrey, cv::COLOR_RGB2GRAY);

    for (int i = 0; i < mLRImage.rows; i++)
    {
        for (int j = 0; j < mLRImage.cols; j++)
        {
            mLRRed.at<cv::Vec3b>(i,j)[1] = 0;
            mLRRed.at<cv::Vec3b>(i,j)[2] = 0;

            mLRGreen.at<cv::Vec3b>(i,j)[0] = 0;
            mLRGreen.at<cv::Vec3b>(i,j)[2] = 0;

            mLRBlue.at<cv::Vec3b>(i,j)[0] = 0;
            mLRBlue.at<cv::Vec3b>(i,j)[1] = 0;
        }
    }
    return true;
}

bool SISR::CreateLRHRPairs()
{
    mRPairs = GetLRHRPairs(mLRRed);
    mGPairs = GetLRHRPairs(mLRGreen);
    mBPairs = GetLRHRPairs(mLRBlue);
    mGreyPairs = GetLRHRPairs(mLRGrey);
    return true;
}



QPixmap SISR::GetStartImage()    // для вывода на qt
{
    return PixmapFromCVMat(mStartImage);
}

QPixmap SISR::GetLRImage()   // для вывода на qt
{
    return PixmapFromCVMat(mLRImage);
}

QPixmap SISR::GetHRImage()   // для вывода на qt
{
    return PixmapFromCVMat(mHRImage);
}

QPixmap SISR::GetLRRed()
{
    return PixmapFromCVMat(mLRRed);
}

QPixmap SISR::GetLRGreen()
{
    return PixmapFromCVMat(mLRGreen);
}

QPixmap SISR::GetLRBlue()
{
    return PixmapFromCVMat(mLRBlue);
}

QPixmap SISR::GetLRGrey()
{
    return PixmapFromCVMatGrey(mLRGrey);
}

int SISR::GetPairsMax()
{
    return mGreyPairs.size();
}

QPair<QPixmap, QPixmap> SISR::GetRPair(int i)
{
    MPair tmp = mRPairs.at(i);
    cv::Mat LR = UpscalePartImage(tmp.first, 50);
    cv::Mat HR = UpscalePartImage(tmp.second, 20);
    return QPair<QPixmap, QPixmap>( PixmapFromCVMat(LR), PixmapFromCVMat(HR) );
}

QPair<QPixmap, QPixmap> SISR::GetGPair(int i)
{
    MPair tmp = mGPairs.at(i);
    cv::Mat LR = UpscalePartImage(tmp.first, 50);
    cv::Mat HR = UpscalePartImage(tmp.second, 20);
    return QPair<QPixmap, QPixmap>( PixmapFromCVMat(LR), PixmapFromCVMat(HR) );
}

QPair<QPixmap, QPixmap> SISR::GetBPair(int i)
{
    MPair tmp = mBPairs.at(i);
    cv::Mat LR = UpscalePartImage(tmp.first, 50);
    cv::Mat HR = UpscalePartImage(tmp.second, 20);
    return QPair<QPixmap, QPixmap>( PixmapFromCVMat(LR), PixmapFromCVMat(HR) );
}

QPair<QPixmap, QPixmap> SISR::GetGreyPair(int i)
{
    MPair tmp = mGreyPairs.at(i);
    cv::Mat LR = UpscalePartImageGrey(tmp.first, 50);
    cv::Mat HR = UpscalePartImageGrey(tmp.second, 20);
    return QPair<QPixmap, QPixmap>( PixmapFromCVMatGrey(LR), PixmapFromCVMatGrey(HR) );
}


QPixmap SISR::PixmapFromCVMat(const cv::Mat& image)
{
    QImage qIm( (uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888 );
    QPixmap pixel;
    pixel.convertFromImage(qIm);
    return pixel;
}

QPixmap SISR::PixmapFromCVMatGrey(const cv::Mat& image)
{
    QImage qIm( (uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_Grayscale8 );
    QPixmap pixel;
    pixel.convertFromImage(qIm);
    return pixel;
}

QList<MPair> SISR::GetLRHRPairs(const cv::Mat& image)
{
    QList<MPair> pairs;
    cv::Mat curImage;
    image.copyTo(curImage);
    while (curImage.rows > 10 && curImage.cols > 10)  // ищем во всех разрешениях, с уменьшением в 0.9 раз
    {
        for (int i = 0; i < curImage.rows - 9; ++i)
        {
            for (int j = 0; j < curImage.cols - 9; ++j)
            {
                cv::Mat HR(curImage, cv::Rect(i, j, 9, 9));
                cv::Mat LR;
                cv::resize(HR, LR, cv::Size(3, 3), cv::INTER_NEAREST);
                pairs.push_back(MPair(LR, HR));
            }
        }
        cv::resize(curImage, curImage, cv::Size(curImage.rows*0.9, curImage.cols*0.9), cv::INTER_NEAREST);
    }
    return pairs;
}

cv::Mat SISR::UpscalePartImage(const cv::Mat& image, int scale)
{
    cv::Mat res;
    cv::resize(image, res, cv::Size(image.rows*scale, image.cols*scale), cv::INTER_NEAREST);
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            res.at<cv::Vec3b>(i,j)[0] = image.at<cv::Vec3b>(i/scale, j/scale)[0];
            res.at<cv::Vec3b>(i,j)[1] = image.at<cv::Vec3b>(i/scale, j/scale)[1];
            res.at<cv::Vec3b>(i,j)[2] = image.at<cv::Vec3b>(i/scale, j/scale)[2];
        }
    }
    return res;
}

cv::Mat SISR::UpscalePartImageGrey(const cv::Mat& image, int scale)
{
    cv::Mat res;
    cv::resize(image, res, cv::Size(image.rows*scale, image.cols*scale), cv::INTER_NEAREST);
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            res.at<uchar>(i,j) = image.at<uchar>(i/scale, j/scale); // серый цвет обходится только так
        }
    }
    return res;
}



