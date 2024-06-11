#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include "ResolutionModel.h"

// std::string model_path = ":/model/model/LSRN.pt";
// std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/LSRN.pt";
// std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/RCAN.pt";
// std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/pytorch_model_4x.pt";
// std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/RCAN_BIX4.pt";
// std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/model_6.pdparams";
// std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/RCAN_BIX4.pdparams";
std::string model_path = "/Users/pmlinelone/Code/CCP3lib/SuperResolution/model/RCAN_BIX2.pdparams";

ResolutionModel Model(model_path);

// copy from internet
cv::Mat qim2mat(QImage & qim)
{
    QImage tmp = qim.copy();
    cv::Mat mat = cv::Mat(tmp.height(), tmp.width(),
                          CV_8UC3,(void*)tmp.constBits(), tmp.bytesPerLine());
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    return mat;
}

// 模型数据转换函数
cv::Mat tensor2Mat(at::Tensor &t)
{
    at::Tensor tmp = t.clone();
    tmp = tmp.squeeze(0).detach().permute({ 1, 2, 0 });
    tmp = tmp.mul(255).clamp(0, 255).to(torch::kU8);
    tmp = tmp.to(torch::kCPU);
    int h_dst = tmp.size(0);
    int w_dst = tmp.size(1);

    cv::Mat mat(h_dst, w_dst, CV_8UC3);
    std::memcpy((void*)mat.data, tmp.data_ptr(), sizeof(torch::kU8) * tmp.numel());
    return mat.clone();
}



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    initImage();
}

QImage MainWindow::MatToQImage(const cv::Mat &mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if (mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for (int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for (int row = 0; row < mat.rows; row++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if (mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if (mat.type() == CV_8UC4)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        return QImage();
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::initImage()
{
    if (oldImage.isNull()) {
        QString filename = ":/base_image/icons/ImageAdd.png";
        ui->imageLabel->setPixmap(QPixmap(filename));
        ui->statusbar->showMessage("请上传图片!");
    }
}


// 上载图片
void MainWindow::on_loadImage_clicked()
{
    QStringList files = QFileDialog::getOpenFileNames(this, "选择文件", "", "Images(*.png *.xpm *.jpg *.bmp)");
    if (files.isEmpty()) return ;

    QString imageFileName =  files.at(0);
    originalPath = imageFileName.toLocal8Bit().constData();
    oldImage.load(imageFileName);
    ui->imageLabel->setPixmap(oldImage);
    ui->actionimageFitH->trigger();
}

// 调整高度
void MainWindow::on_actionimageFitH_triggered()
{
    if (ui->switchSizeButton->text() == "原始大小") {
        int h = ui->imageLabel->height();
        // originalHeight = oldImage.height();  // 理论上不需要该 parameter
        QPixmap resizePix = ui->imageLabel->pixmap().scaledToHeight(h - 30);
        ui->imageLabel->setPixmap(resizePix);
    }
}


void MainWindow::on_switchSizeButton_clicked()
{
    if (oldImage.isNull()) return ;
    QString text = ui->switchSizeButton->text();
    if (text == "原始大小") {
        if (newImage.isNull()) {
            ui->imageLabel->setPixmap(oldImage);
        } else {
            ui->imageLabel->setPixmap(newImage);
        }

        ui->switchSizeButton->setText("自适应");
    } else if (text == "自适应") {
        ui->switchSizeButton->setText("原始大小");
        ui->actionimageFitH->trigger();
    }
}

// 保存图片
void MainWindow::on_saveImageButton_clicked()
{
    if (oldImage.isNull() || newImage.isNull()) {
        ui->statusbar->showMessage("图片未上传或未进行超分辨率。");
        return ;
    }

    QImage superImage(newImage.toImage());

    QString filename = QFileDialog::getSaveFileName(this, tr("保存图片"), "", "*.png;; *.jpg;; *.bmp;; *.tif;; *.GIF"); //选择路径

    if (filename.isEmpty()) return ;

    if (superImage.save(filename)) {
        ui->statusbar->showMessage("保存成功");
    }
}


void MainWindow::on_superResolutionButton_clicked()
{
    ui->statusbar->showMessage("执行超分辨率!");
    if (oldImage.isNull()) {
        QMessageBox::warning(nullptr, "提示", "请先选择一张图片！", QMessageBox::Yes |  QMessageBox::Yes);
        return ;
    }
    std::cout << "loading!" << std::endl;
    img.load(originalPath);
    std::cout << "loading后!" << std::endl;
    // img.imshow("img");
    at::Tensor feature = img.feature.clone();
    std::cout << "forward中!" << std::endl;
    at::Tensor output = Model.forward(feature);
    std::cout << "forward已经完成!" << std::endl;
    sr_img = tensor2Mat(output);
    cv::imshow("superResolution", sr_img);
    cv::Mat upscale = sr_img.clone();
    QImage image2 = MatToQImage(upscale);
    newImage = QPixmap::fromImage(image2);
    ui->imageLabel->setPixmap(newImage);
    ui->actionimageFitH->trigger();
    return ;
}

