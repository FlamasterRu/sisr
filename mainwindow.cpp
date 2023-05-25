#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // настройки
    ui->pushButtonSwipeLeft->setVisible(false);
    ui->pushButtonSwipeRight->setVisible(false);
    ui->labelChoosedImageNum->setVisible(false);
    ui->pushButtonCount->setVisible(false);

    ui->comboBoxScalling->setVisible(false);
    ui->comboBoxScalling->clear();
    ui->comboBoxScalling->addItems(QStringList() << QString("2.0") << QStringList("4.0"));

    ui->comboBoxImageType->setVisible(false);
    ui->comboBoxImageType->clear();
    ui->comboBoxImageType->addItems(QStringList() << QString("Grey"));

    DefaultTab();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButtonFilePath_clicked()
{
    // путь к папке с файлами
//    QString dirPath = QFileDialog::getExistingDirectory(this, QString("Укажите директорию"));
//    if (dirPath.size() == 0)
//    {
//        return; // ничего не указали
//    }
    //QString dirPath("D:/nikita_files/nngu/diplom/sisr_images");
    QString dirPath("D:/projects/other/HR");

    // получение имён всех изображений
    QDir dir(dirPath);
    mStartImageFileNames = dir.entryList(QStringList() << "*.jpg" << "*.png");  // фильтры по расширениям файлов
    if (mStartImageFileNames.size() == 0)
    {
        return; // в папке нет картинок
    }
    mStartImageFileNames.sort();    // сортировка по алфавиту
    for (QString& fileName : mStartImageFileNames)
    {
        fileName = dir.absoluteFilePath(fileName);
    }
    ui->lineEditDirPath->setText(dirPath);

    // включение кнопок переключения изображения
    ui->pushButtonSwipeLeft->setVisible(true);
    ui->pushButtonSwipeRight->setVisible(true);
    ui->labelChoosedImageNum->setVisible(true);
    ui->labelChoosedImageNum->setText(QString("0/") + QString::number(mStartImageFileNames.size()));
    ui->comboBoxScalling->setVisible(true);
    ui->comboBoxImageType->setVisible(true);
    ui->pushButtonCount->setVisible(true);

    InitStartImage(0);
}

void MainWindow::on_pushButtonCount_clicked()
{
    DefaultTab();
    if (ui->comboBoxImageType->currentText() == QString("Grey"))
    {
        // вторая вкладка (исходное изображение)
        cv::Mat LRGreyImage, HRGreyImage;
        cv::cvtColor(mStartImage, HRGreyImage, cv::COLOR_RGB2GRAY);
        ui->Image1->setPixmap(PixmapFromCVMat(HRGreyImage, QImage::Format_Grayscale8));
        cv::cvtColor(mLRImage, LRGreyImage, cv::COLOR_RGB2GRAY);
        ui->Image2->setPixmap(PixmapFromCVMat(LRGreyImage, QImage::Format_Grayscale8));

        // третья вкладка (пары патчей)
        s1.InitImage(LRGreyImage);
        s1.CreateLRHRPairs();

        ui->label1PartCur->setVisible(true);
        ui->label1PartMax->setVisible(true);
        ui->horizontalSlider1Part->setVisible(true);
        ui->label1PartCur->setText(0);
        ui->label1PartMax->setText(QString::number(s1.GetPairsCount()-1));
        ui->horizontalSlider1Part->setMaximum(s1.GetPairsCount()-1);
        ui->horizontalSlider1Part->setValue(1); // костыль, чтобы картинки обновились
        ui->horizontalSlider1Part->setValue(0); // костыль, чтобы картинки обновились
    }
}

void MainWindow::on_pushButtonSwipeLeft_clicked()
{
    QStringList tmp = ui->labelChoosedImageNum->text().split(QString("/"));
    int currentImageNum = tmp.first().toInt(); // номер текущей картинки
    if (currentImageNum <= 0)
    {
        return; // влево нет цифр
    }

    ui->labelChoosedImageNum->setText(QString::number(currentImageNum-1) + QString("/") + QString::number(mStartImageFileNames.size()));
    InitStartImage(currentImageNum-1);
}

void MainWindow::on_pushButtonSwipeRight_clicked()
{
    QStringList tmp = ui->labelChoosedImageNum->text().split(QString("/"));
    int currentImageNum = tmp.first().toInt(); // номер текущей картинки
    if (currentImageNum >= mStartImageFileNames.size()-1)
    {
        return; // влево нет цифр
    }

    ui->labelChoosedImageNum->setText(QString::number(currentImageNum+1) + QString("/") + QString::number(mStartImageFileNames.size()));
    InitStartImage(currentImageNum+1);
}

void MainWindow::on_horizontalSlider1Part_valueChanged(int value)
{
    if (mStartImageFileNames.isEmpty())
    {
        return;
    }
    MPair p = s1.GetPair(value);
    ui->RLRPart->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(p.first, 50), QImage::Format_Grayscale8));
    ui->RHRPart->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(p.second, 20), QImage::Format_Grayscale8));
    ui->label1PartCur->setText(QString::number(value));
}

void MainWindow::on_comboBoxScalling_currentTextChanged(const QString &arg1)
{
    if (mStartImageFileNames.isEmpty())
    {
        return;
    }

    // создаёт LR изображение, в зависимости от выбранной настройки
    int scalling = 2;
    if (ui->comboBoxScalling->currentText() == QString("2.0"))
    {
        scalling = 2;
    }
    else if (ui->comboBoxScalling->currentText() == QString("4.0"))
    {
        scalling = 4;
    }
    cv::resize(mStartImage, mLRImage, cv::Size(mStartImage.cols/scalling, mStartImage.rows/scalling), cv::INTER_CUBIC);
    ui->LRImage->setPixmap(PixmapFromCVMat(mLRImage, QImage::Format_RGB888));
}


void MainWindow::DefaultTab()
{
    // вкладка LR-HR пары
    ui->label1PartCur->setVisible(false);
    ui->label1PartMax->setVisible(false);
    ui->horizontalSlider1Part->setVisible(false);
    ui->label2PartCur->setVisible(false);
    ui->label2PartMax->setVisible(false);
    ui->horizontalSlider2Part->setVisible(false);
    ui->label3PartCur->setVisible(false);
    ui->label3PartMax->setVisible(false);
    ui->horizontalSlider3Part->setVisible(false);
    ui->label4PartCur->setVisible(false);
    ui->label4PartMax->setVisible(false);
    ui->horizontalSlider4Part->setVisible(false);
}

void MainWindow::InitStartImage(const int imageNum)
{
    mStartImage = cv::imread(mStartImageFileNames[imageNum].toStdString()); // чтение изображения
    cv::cvtColor(mStartImage, mStartImage, cv::COLOR_BGR2RGB);  // преобразует к нормальной rgb палитре (читает в bgr)
    ui->startImage->setPixmap(PixmapFromCVMat(mStartImage, QImage::Format_RGB888));
    on_comboBoxScalling_currentTextChanged(ui->comboBoxScalling->currentText());
}

QPixmap MainWindow::PixmapFromCVMat(const cv::Mat& image, QImage::Format format)
{
    QImage qIm( (uchar*)image.data, image.cols, image.rows, image.step, format );
    QPixmap pixel;
    pixel.convertFromImage(qIm);
    return pixel;
}

cv::Mat MainWindow::UpscalePartImage(const cv::Mat& image, int scale)
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

cv::Mat MainWindow::UpscalePartImageGrey(const cv::Mat& image, int scale)
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






