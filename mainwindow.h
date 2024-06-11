#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include "Image.h"

class QFileDialog;
class QPixmap;
class QMessageBox;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    QImage MatToQImage(const cv::Mat& mat);
    ~MainWindow();

private slots:
    void on_loadImage_clicked();

    void on_actionimageFitH_triggered();

    void on_switchSizeButton_clicked();

    void on_saveImageButton_clicked();

    void on_superResolutionButton_clicked();

private:
    Ui::MainWindow *ui;

    // 图片处理
    Image img;
    cv::Mat sr_img;

    // 存放图片的QPixmap对象
    int originalHeight;
    int originalWidth;
    std::string originalPath;

    QPixmap oldImage;
    QPixmap newImage;

    void initImage();
};
#endif // MAINWINDOW_H
