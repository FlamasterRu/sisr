#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_mainwindow.h"
#include <sisr.h>

#include <iostream>
#include <QFileDialog>
#include <QDir>
#include <QKeyEvent>



namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    //void keyPressEvent(QKeyEvent *e);

private slots:
    void on_pushButtonFilePath_clicked();

    void on_pushButtonSwipeLeft_clicked();

    void on_pushButtonSwipeRight_clicked();

private:
    void InitStartImage(const int imageNum);

private:
    Ui::MainWindow *ui;

    QStringList mStartImageFileNames;
    SISR mSISR;
};

#endif // MAINWINDOW_H
