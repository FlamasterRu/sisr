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

    // вкладка LR-HR пары
    ui->horizontalSliderRPart->setVisible(false);
    ui->horizontalSliderGPart->setVisible(false);
    ui->horizontalSliderBPart->setVisible(false);
    ui->horizontalSliderGreyPart->setVisible(false);
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
    QString dirPath("D:/nikita_files/nngu/diplom/sisr_images");

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
    ui->pushButtonCount->setVisible(true);
    ui->labelChoosedImageNum->setText(QString("0/") + QString::number(mStartImageFileNames.size()));

    InitStartImage(0);
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

void MainWindow::InitStartImage(const int imageNum)
{
    mSISR.InitImage(mStartImageFileNames[imageNum]);    // инициализация расчётов

    ui->startImage->setPixmap(mSISR.GetStartImage());
    ui->LRImage->setPixmap(mSISR.GetLRImage());
}

void MainWindow::on_pushButtonCount_clicked()
{
    if (mStartImageFileNames.isEmpty())
    {
        return;
    }

    // разбиваем изображение на цветовые каналы
    mSISR.SplitColors();
    ui->RColorLRImage->setPixmap(mSISR.GetLRRed());
    ui->GColorLRImage->setPixmap(mSISR.GetLRGreen());
    ui->BColorLRImage->setPixmap(mSISR.GetLRBlue());
    ui->GreyLRImage->setPixmap(mSISR.GetLRGrey());

    // получаем пары LR-HR частей изображения
    mSISR.CreateLRHRPairs();

    ui->horizontalSliderRPart->setVisible(true);
    ui->horizontalSliderGPart->setVisible(true);
    ui->horizontalSliderBPart->setVisible(true);
    ui->horizontalSliderGreyPart->setVisible(true);

    ui->labelRPartMax->setText(QString::number(mSISR.GetPairsMax()) + QString("  |"));
    ui->labelGPartMax->setText(QString::number(mSISR.GetPairsMax()) + QString("  |"));
    ui->labelBPartMax->setText(QString::number(mSISR.GetPairsMax()) + QString("  |"));
    ui->labelGreyPartMax->setText(QString::number(mSISR.GetPairsMax()));

    ui->horizontalSliderRPart->setMaximum(mSISR.GetPairsMax());
    ui->horizontalSliderGPart->setMaximum(mSISR.GetPairsMax());
    ui->horizontalSliderBPart->setMaximum(mSISR.GetPairsMax());
    ui->horizontalSliderGreyPart->setMaximum(mSISR.GetPairsMax());

    ui->horizontalSliderRPart->setValue(1); // костыль, чтобы картинки обновились
    ui->horizontalSliderGPart->setValue(1);
    ui->horizontalSliderBPart->setValue(1);
    ui->horizontalSliderGreyPart->setValue(1);
    ui->horizontalSliderRPart->setValue(0);
    ui->horizontalSliderGPart->setValue(0);
    ui->horizontalSliderBPart->setValue(0);
    ui->horizontalSliderGreyPart->setValue(0);
}

void MainWindow::on_horizontalSliderRPart_valueChanged(int value)
{
    if (mStartImageFileNames.isEmpty() || value >= mSISR.GetPairsMax())
    {
        return;
    }
    ui->RLRPart->setPixmap(mSISR.GetRPair(value).first);
    ui->RHRPart->setPixmap(mSISR.GetRPair(value).second);
    ui->labelRPartCur->setText(QString::number(value));
}

void MainWindow::on_horizontalSliderGPart_valueChanged(int value)
{
    if (mStartImageFileNames.isEmpty() || value >= mSISR.GetPairsMax())
    {
        return;
    }
    ui->GLRPart->setPixmap(mSISR.GetGPair(value).first);
    ui->GHRPart->setPixmap(mSISR.GetGPair(value).second);
    ui->labelGPartCur->setText(QString::number(value));
}

void MainWindow::on_horizontalSliderBPart_valueChanged(int value)
{
    if (mStartImageFileNames.isEmpty() || value >= mSISR.GetPairsMax())
    {
        return;
    }
    ui->BLRPart->setPixmap(mSISR.GetBPair(value).first);
    ui->BHRPart->setPixmap(mSISR.GetBPair(value).second);
    ui->labelBPartCur->setText(QString::number(value));
}

void MainWindow::on_horizontalSliderGreyPart_valueChanged(int value)
{
    if (mStartImageFileNames.isEmpty() || value >= mSISR.GetPairsMax())
    {
        return;
    }
    ui->GreyLRPart->setPixmap(mSISR.GetGreyPair(value).first);
    ui->GreyHRPart->setPixmap(mSISR.GetGreyPair(value).second);
    ui->labelGreyPartCur->setText(QString::number(value));
}
