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
#include <QTime>

#define C1 (float) (0.01 * 255 * 0.01  * 255)
#define C2 (float) (0.03 * 255 * 0.03  * 255)

typedef QPair<cv::Mat, cv::Mat> PairImag;

// LR - фрагмент исходного изображения низкого разрешения
// HR - фрагмент исходного изображения высокого разрешения
// AHR - построенный фрагмент высокого разрешения

/// \brief Хранит пару фрагментов низкого и высокого разрешения. И данные для отображения в gui.
struct LRAHRInfo
{
    cv::Mat LR, HR;
    cv::Rect rect;
    QList<int> patchNums;
    QList<double> distToPatches;
};

/// \brief Класс содержащий вариации алгоритма повышения разрешения одиночного изображения.
class SISR
{
public:
    SISR();

    /// \brief Задаёт начальное изображение низкого разрешения.
    bool InitImage(const cv::Mat& image);

    /// \brief Строит список пар патчей и хэш таблицу для быстрого поиска.
    bool CreateLRHRPairs();

    /// \brief Строит изображение высокого разрешения.
    bool AssemblyHRImage();

    /// \brief Возвращает размер списока пар фрагментов низкого и высокого разрешения из исходного изображения.
    int GetPairsCount();
    /// \brief Возвращает пару фрагментов низкого и высокого разрешения из исходного изображения.
    PairImag GetPair(int i);

    /// \brief Возвращает размер списка патчей.
    int GetPatchesCount();
    /// \brief Возвращает объект из LR, AHR и дополнительной информации по сборке AHR.
    LRAHRInfo GetPatch(int i);

    /// \brief Возвращает изображение высокого разрешения.
    cv::Mat GetHRImage();

    /// \brief Среднее арифметическое по пикселям.
    static double Avg(const cv::Mat& m);
    /// \brief Среднеквадратичное отклонение от Avg
    static double Sigma(const cv::Mat& m);
    /// \brief Ковариация
    static double Cov(const cv::Mat& m1, const cv::Mat& m2);
    /// \brief Евклидово расстояние (корень от сумму квадратов расстояний между пикселями). Чем меньше - тем сильнее похожи.
    static double EuclidDist(const cv::Mat& i1, const cv::Mat& i2);
    /// \brief Среднее из квадратов отклонений между пикселями. Чем меньше - тем сильнее похожи.
    static double StandartDerivation(const cv::Mat& i1, const cv::Mat& i2);
    /// \brief Среднеквадратичное отклонение (корень из стандартного отклонения). Чем меньше - тем сильнее похожи.
    static double RMSE(const cv::Mat& image1, const cv::Mat& image2);
    /// \brief Максимальное отклонение пикселя (модуль). Чем меньше - тем сильнее похожи.
    static double MaxDeviation(const cv::Mat& image1, const cv::Mat& image2);
    /// \brief Отношение среднеквадратичного отклонения к максимальному значению в пикселях. Наилучшее значение в интервале от 30 до 40 дБ.
    static double PSNR(const cv::Mat& image1, const cv::Mat& image2);
    /// \brief Индекс структурного сходства (учитывает взаимосвязь соседних пикселей). Значение от -1, до 1. Чем ближе к 1 - тем сильнее похожи.
    static double SSIM(const cv::Mat& image1, const cv::Mat& image2);
    /// \brief Структурные отличия (обратное к (1 - SSIM)/2). Значение от 0 до 1. Чем меньше - тем сильнее похожи.
    static double DSSIM(const cv::Mat& image1, const cv::Mat& image2);
    /// \brief Максимальное значение пикселя.
    static double Max(const cv::Mat& image);


private:
    // Поиск подходящих пар LR-HR для LR
    /// \brief Простой поиск по списку за О(n).
    void GetNearestPairsIDS(const cv::Mat& part, QList<int>& nearest, QList<double>& dist);
    /// \ brief Поиск в хэш таблице за О(1).
    void GetNearestPairsIDS2(const cv::Mat& part, QList<int>& nearest, QList<double>& dist);

    /// \brief Сборка одного патча высокого разрешения из нескольких подходящих.
    cv::Mat AssemblyHRPatch(QList<int> nearest);

private:
    /// \brief Изображения низкого и увеличенного разрешения
    cv::Mat mLRImage, mHRImage;

    /// \brief Список пар фрагментов низкого и высокого разрешения.
    QList<PairImag> mPairs;

    /// \brief Таблица пар для быстрого поиска похожих
    QHash<uint, QList<int>> mHash;

    /// \brief Список фрагментов низкого разрешения и полученных фрагментов высокого разрешения.
    QList<LRAHRInfo> mPatches;

    double sumT1, sumT2;
};

#endif // SISR_H
