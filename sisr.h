#ifndef SISR_H
#define SISR_H

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QList>
#include <QMap>
#include <QPair>
#include <omp.h>

#define MPair QPair<cv::Mat, cv::Mat>

struct Patch
{
    cv::Mat LR, HR;
    cv::Rect rect;
    QList<int> patchNums;
    QList<double> distToPatches;
};

class SISR  // 5, 61, 78
{
public:
    SISR();

    bool InitImage(const cv::Mat& image);    // инициализация начального изображения
    bool CreateLRHRPairs(); // для всех изображений строит пары мелких, крупных фрагментов
    bool AssemblyHRImage(); // сборка HR изображения из пар патчей

    int GetPairsCount();
    MPair GetPair(int i);

    int GetPatchesCount();
    Patch GetPatch(int i);

    cv::Mat GetHRImage();

    static double RMSE(const cv::Mat& image1, const cv::Mat& image2);
    static double MaxDeviation(const cv::Mat& image1, const cv::Mat& image2);
    static double PSNR(const cv::Mat& image1, const cv::Mat& image2);
    static double SSIM(const cv::Mat& image1, const cv::Mat& image2);
    static double Max(const cv::Mat& image);

private:
    void GetNearestPairsIDS(const cv::Mat& part, QList<int>& nearest, QList<double>& dist);
    double EuclidDist(const cv::Mat& i1, const cv::Mat& i2);
    double StandartDerivation(const cv::Mat& i1, const cv::Mat& i2);
    cv::Mat AssemblyHRPatch(QList<int> nearest, QList<double> weight);

private:
    cv::Mat mLRImage, mHRImage;
    QList<MPair> mPairs;
    QList<Patch> mPatches;
};

#endif // SISR_H
