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

#define MPair QPair<cv::Mat, cv::Mat>

class SISR
{
public:
    SISR();

    bool InitImage(const cv::Mat& image);    // инициализация начального изображения
    //bool SplitColors(); // разобъёт LR изображение по цветам
    bool CreateLRHRPairs(); // для всех изображений строит пары мелких, крупных фрагментов

    int GetPairsCount();
    MPair GetPair(int i);


private:
    cv::Mat mLRImage, mHRImage;
    QList<MPair> mPairs;
};

#endif // SISR_H
