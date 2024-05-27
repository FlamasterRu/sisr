#include "sisr.h"

uchar hashUnity = 1;

cv::Mat VaveletHaaraMain(const cv::Mat& image)
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

cv::Mat VaveletHaara(const cv::Mat& image)
{
    if (image.rows != image.cols)
        throw("image.rows != image.cols");

    cv::Mat res1(image.rows, image.cols, CV_8SC1);
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols/2; ++j)
        {
            int v1 = image.at<uchar>(i, 2*j);
            int v2 = image.at<uchar>(i, 2*j+1);
            res1.at<char>(i, j) = (v1 + v2) / 2;
            res1.at<char>(i, image.cols/2 + j) = (v1 - v2) / 2;
        }
    }

    cv::Mat res2(image.rows, image.cols, CV_8SC1);
    for (int i = 0; i < image.rows/2; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            int v1 = res1.at<char>(2*i, j);
            int v2 = res1.at<char>(2*i+1, j);
            res2.at<char>(i, j) = (v1 + v2) / 2;
            res2.at<char>(image.rows/2 + i, j) = (v1 - v2) / 2;
        }
    }
    return res2;
}

cv::Mat ReVaveletHaara(const cv::Mat& image)
{
    if (image.rows != image.cols)
        throw("image.rows != image.cols");

    cv::Mat res1(image.rows, image.cols, CV_8SC1);
    for (int i = 0; i < image.rows/2; ++i)
    {
        for (int j = 0; j < image.cols; ++j)
        {
            double v1 = image.at<char>(i, j);
            double v2 = image.at<char>(image.rows/2 + i, j);
            res1.at<char>(2*i, j) = (v1 - v2) * 2;
            res1.at<char>(2*i + 1, j) = (v1 + v2) * 2;
        }
    }

    cv::Mat res2(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; ++i)
    {
        for (int j = 0; j < image.cols/2; ++j)
        {
            double v1 = res1.at<char>(i, j);
            double v2 = res1.at<char>(i, image.cols/2 + j);
            res2.at<uchar>(i, 2*j) = (v1 - v2) * 2;
            res2.at<uchar>(i, 2*j + 1) = (v1 + v2) * 2;
        }
    }

    return res2;
}

uint qHash(const cv::Mat& image)
{
    cv::Mat cop(image);
    while (cop.rows * cop.cols > 4)
    {
        cop = VaveletHaaraMain(cop);
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


int SISR::GetPairsCount()
{
    return mPairs.size();
}

PairImag SISR::GetPair(int i)
{
    return mPairs.at(i);
}

int SISR::GetPatchesCount()
{
    return mPatches.size();
}

LRAHRInfo SISR::GetPatch(int i)
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

bool SISR::CreateLRHRPairs(bool save)
{
    // Чтобы уменьшать размер внутри объекта
    cv::Mat curImage;
    mLRImage.copyTo(curImage);

    // Строит пары, пока изображение больше чем 9*9 пикселей
    while (curImage.rows >= mHrPatchSize && curImage.cols >= mHrPatchSize)
    {
        for (int i = 0; i <= curImage.rows - mHrPatchSize; ++i)
        {
            for (int j = 0; j <= curImage.cols - mHrPatchSize; ++j)
            {
                // Фрагмент высокого разрешения
                cv::Mat HR(curImage, cv::Rect(j, i, mHrPatchSize, mHrPatchSize));

                // Фрагмент низкого разрешения
                cv::Mat LR;
                cv::resize(HR, LR, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LR, HR));

                // Зеркально отображены
                cv::Mat HRZ;
                cv::flip(HR, HRZ, 1);
                cv::Mat LRZ;
                cv::resize(HRZ, LRZ, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LRZ, HRZ));

                // Поворот на 90 градусов
                cv::Mat HR90;
                cv::rotate(HR, HR90, cv::ROTATE_90_CLOCKWISE);
                cv::Mat LR90;
                cv::resize(HR90, LR90, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LR90, HR90));

                // Поворот на 90 и зеркально
                cv::Mat HRZ90;
                cv::flip(HR90, HRZ90, 1);
                cv::Mat LRZ90;
                cv::resize(HRZ90, LRZ90, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LRZ90, HRZ90));

                // Поворот на 180 градусов
                cv::Mat HR180;
                cv::rotate(HR, HR180, cv::ROTATE_180);
                cv::Mat LR180;
                cv::resize(HR180, LR180, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LR180, HR180));

                // Поворот на 180 и зеркально
                cv::Mat HRZ180;
                cv::flip(HR180, HRZ180, 1);
                cv::Mat LRZ180;
                cv::resize(HRZ180, LRZ180, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LRZ180, HRZ180));

                // Поворот на 270 градусов
                cv::Mat HR270;
                cv::rotate(HR, HR270, cv::ROTATE_90_COUNTERCLOCKWISE);
                cv::Mat LR270;
                cv::resize(HR270, LR270, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LR270, HR270));

                // Поворот на 270 и зеркально
                cv::Mat HRZ270;
                cv::flip(HR270, HRZ270, 1);
                cv::Mat LRZ270;
                cv::resize(HRZ270, LRZ270, cv::Size(mLrPatchSize, mLrPatchSize), cv::INTER_NEAREST);
                mPairs.push_back(PairImag(LRZ270, HRZ270));
            }
        }
        cv::resize(curImage, curImage, cv::Size(curImage.rows*0.5, curImage.cols*0.5), cv::INTER_NEAREST);
    }

    std::cout << "Hash teor size = " << mPairs.size() << std::endl;
    if (mPairs.size() < 16)
        hashUnity = 128;
    else if (mPairs.size() < 256)
        hashUnity = 128;
    else if (mPairs.size() < 4096)
        hashUnity = 64;
    else if (mPairs.size() < 65536)
        hashUnity = 32;
    else if (mPairs.size() < 1048576)
        hashUnity = 16;
    else if (mPairs.size() < 16777216)
        hashUnity = 16;
    else if (mPairs.size() < 268435456)
        hashUnity = 8;
    else
        hashUnity = 4;
    hashUnity = 16;
    std::cout << "Hash unity = " << (uint)hashUnity << std::endl;

    if (save)
    {
        QDir lrDir, hrDir;
        lrDir.mkdir("lr");
        lrDir.cd("lr");
        hrDir.mkdir("hr");
        hrDir.cd("hr");
        for (int i = 0; i < mPairs.size(); ++i)
        {
            const PairImag& p = mPairs[i];
            cv::Mat lr = p.first;
            cv::imwrite(lrDir.absolutePath().toStdString() + "/" + std::to_string(i) + ".jpg", lr);
            cv::Mat hr = p.second;
            cv::imwrite(hrDir.absolutePath().toStdString() + "/" + std::to_string(i) + ".jpg", hr);
            mHash[qHash(p.first)].push_back(i);
        }
    }
    std::cout << mPairs.size() << std::endl;

    return true;
}

bool SISR::AssemblyHRImage(bool save)
{
    QList<int> nearestPairs;
    QList<double> dist;
    // Заполнение серым для контраста результата.
    cv::Mat tmpHR;
    tmpHR.create(mLRImage.rows*2, mLRImage.cols*2, CV_64F);
    for (int i = 0; i < tmpHR.rows; ++i)
    {
        for (int j = 0; j < tmpHR.cols; ++j)
        {
            tmpHR.at<double>(i,j) = 0;
        }
    }

    if (save)
    {
        QDir lr_f;
        lr_f.mkdir("lr_f");
        lr_f.mkdir("hr_f");
        lr_f.cd("lr_f");
        for (int i = 0; i <= mLRImage.rows - mLrPatchSize; i += 4)
        {
            for (int j = 0; j <= mLRImage.cols - mLrPatchSize; j += 4)
            {
                cv::Mat LRpart(mLRImage, cv::Rect(j, i, mLrPatchSize, mLrPatchSize));
                std::string path = lr_f.absolutePath().toStdString() + "/" + std::to_string(i) + "_" + std::to_string(j) + ".jpg";
                cv::imwrite(path, LRpart);
            }
        }

        lr_f.cdUp();
        QStringList args;
        args << lr_f.absolutePath() + "/dm.py" << "--dataset" << lr_f.absolutePath();
        python = std::make_unique<QProcess>(this);
        connect(python.get(), &QProcess::readyReadStandardOutput, this, &SISR::ReadPythonOutput);
        connect(python.get(), &QProcess::readyReadStandardError, this, &SISR::ReadPythonErrors);
        python->start("python", args);
        while (!python->waitForFinished(100))
        {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 100);
        }
    }


    sumT1 = 0;
    sumT2 = 0;

    for (int i = 0; i <= mLRImage.rows - mLrPatchSize; i += 4)
    {
        for (int j = 0; j <= mLRImage.cols - mLrPatchSize; j += 4)
        {
            cv::Mat LRpart(mLRImage, cv::Rect(j, i, mLrPatchSize, mLrPatchSize));
            GetNearestPairsIDS2(LRpart, nearestPairs, dist);

            // Построение фрагмента высокого разрешения.
            LRAHRInfo p;
            p.rect = cv::Rect(j, i, mLrPatchSize, mLrPatchSize);
            p.patchNums = nearestPairs;
            p.distToPatches = dist;
            p.LR = LRpart;
            if (save)
            {
                QDir hr_f;
                hr_f.cd("hr_f");
                cv::Mat sr3;
                sr3.create(16, 16, CV_8U);
                for (int i1 = 0; i1 < sr3.rows; ++i1)
                {
                    for (int j1 = 0; j1 < sr3.cols; ++j1)
                    {
                        sr3.at<uchar>(i1, j1) = 0;
                    }
                }
                QString path = hr_f.absolutePath() + "/" + QString::number(i) + "_" + QString::number(j) + ".jpg";
                QFileInfo info(path);
                if (info.exists())
                {
                    sr3 = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE ); // чтение изображения
                }
                p.HR = sr3;
            }
            else
            {
                p.HR = AssemblyHRPatch(nearestPairs);
            }
            mPatches.push_back(p);

            for (int ih = 2*i, ip = 0; ip < p.HR.rows; ++ih, ++ip)
            {
                for (int jh = 2*j, jp = 0; jp < p.HR.cols; ++jh, ++jp)
                {
                    double hv = tmpHR.at<double>(ih, jh);
                    double pv = p.HR.at<uchar>(ip, jp);
                    if (std::abs(hv) < 1e-5)
                        tmpHR.at<double>(ih, jh) = pv;
                    else
                        tmpHR.at<double>(ih, jh) = (hv + pv) / 2.;
                }
            }

        }
    }
    tmpHR.convertTo(mHRImage, CV_8U);

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
    // Сортировка первых четырёх пар по расстоянию.
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
        for (uint idx : mHash[qHash(part)])
        {
            double d = EuclidDist(part, mPairs[idx].first);
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
        GetNearestPairsIDS(part, nearest, dist);
    }
    sumT1 += t.restart();
}


cv::Mat SISR::AssemblyHRPatch(QList<int> nearest)
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

double SISR::Avg(const cv::Mat& m)
{
    double res = 0.;
    for (int i = 0; i < m.rows; ++i)
    {
        for (int j = 0; j < m.cols; ++j)
        {
            res += m.at<uchar>(i,j);
        }
    }
    return res/static_cast<double>(m.rows*m.cols);
}

double SISR::Sigma(const cv::Mat& m)
{
    double mu = Avg(m);
    double res = 0;
    for (int i = 0; i < m.rows; ++i)
    {
        for (int j = 0; j < m.cols; ++j)
        {
            double v = m.at<uchar>(i,j);
            res += std::abs( (v-mu)*(v-mu) );
        }
    }
    return res / static_cast<double>(m.rows*m.cols);
}

double SISR::Cov(const cv::Mat& m1, const cv::Mat& m2)
{
    double res = 0;
    double mu1 = Avg(m1);
    double mu2 = Avg(m2);
    for (int i = 0; i < m1.rows; ++i)
    {
        for (int j = 0; j < m1.cols; ++j)
        {
            double v1 = m1.at<uchar>(i,j);
            double v2 = m2.at<uchar>(i,j);
            res += (v1 - mu1)*(v2 - mu2);
        }
    }
    return res / static_cast<double>(m1.rows*m1.cols);
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
    return res/static_cast<double>(i1.rows*i2.cols);
}

double SISR::RMSE(const cv::Mat& image1, const cv::Mat& image2)
{
    return std::sqrt( StandartDerivation(image1, image2) );
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
    double mu1 = Avg(image1);
    double mu2 = Avg(image2);
    double sigma1 = Sigma(image1);
    double sigma2 = Sigma(image2);
    double cov = Cov(image1, image2);
    return (2.*mu1*mu2 + C1)*(2.*cov + C2) / ( (mu1*mu1 + mu2*mu2 + C1)*(sigma1*sigma1 + sigma2*sigma2 + C2) );
}

double SISR::DSSIM(const cv::Mat& image1, const cv::Mat& image2)
{
    return (1. - SSIM(image1, image2))/2.;
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

void SISR::ReadPythonOutput()
{
    QString tmp  =python->readAllStandardOutput();
    emit Message(tmp);
}

void SISR::ReadPythonErrors()
{
    QString tmp = python->readAllStandardError();
    emit Message(tmp);
}
