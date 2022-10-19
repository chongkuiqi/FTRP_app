#include "util.h"
#include "probability.h"
#include "ui_probability.h"
#include <QFileDialog>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <string>

#include <QDebug>
#include <cmath>
#include "roi_align_rotated_cpu.h"

using std::cout;
using std::endl;

Probability::Probability(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Probability)
{
    ui->setupUi(this);
    connect(ui->bu_exit, &QPushButton::clicked, this, &QMainWindow::close);
    // 读取图像文件路径
    connect(ui->bu_browse, &QPushButton::clicked, this, &Probability::browse_img);

    // 读取模型文件路径
    connect(ui->bu_browse_gt, &QPushButton::clicked, this, &Probability::browse_gt);

    // 提取特征
    connect(ui->bu_extract, &QPushButton::clicked, this, &Probability::extract_fe);


    connect(ui->bu_similarity, &QPushButton::clicked, this, &Probability::cal_similarity);

    connect(ui->bu_probability, &QPushButton::clicked, this, &Probability::cal_probability);

    connect(ui->bu_save, &QPushButton::clicked, this, &Probability::save_results);
    // 显示软件日志
    ui->text_log->setText("请输入图像路径和标注文件路径...");

    // 保存路径
    connect(ui->bu_browse_save, &QPushButton::clicked, this, &Probability::browse_save);

    // 设置label的大小
    ui->labelImage->resize(512,512);
    // 图像自适应label的大小
    ui->labelImage->setScaledContents(true);



    // 三种图像特征选择按钮，放进一个按钮组
    bu_fe_group.setParent(this);
    // 按钮组中不互斥
    bu_fe_group.setExclusive(false);
    bu_fe_group.addButton(ui->bu_deep_fe, 0);
    bu_fe_group.addButton(ui->bu_gray_fe, 1);
    bu_fe_group.addButton(ui->bu_text_fe, 2);

    // 按钮组每改变一次状态，都会调用一次
    connect(&bu_fe_group, QOverload<QAbstractButton *>::of(&QButtonGroup::buttonClicked), this, &Probability::on_bu_fe_group);

    fe_status = {false, false, false};


}

Probability::~Probability()
{
    delete ui;
}


void Probability::browse_img()
{
    QString img_path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Images(*.png *.jpg)"
                );

    ui->le_imgpath->setText(img_path);

    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(img_path);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg->scaled(ui->labelImage->size(),Qt::KeepAspectRatio);
    ui->labelImage->setPixmap(QPixmap::fromImage(dest));

}

void Probability::browse_gt()
{
    QString path = QFileDialog::getOpenFileName(
                this,
                "open",
                "../",
                "Text files(*.txt)"
                );

    ui->le_gtpath->setText(path);
}



void Probability::browse_save()
{
    QString path = QFileDialog::getExistingDirectory(this,"open","../");
    // qDebug() << path;
    ui->le_savepath->setText(path);
}


void Probability::extract_fe()
{
    QString img_path = ui->le_imgpath->text();

    LoadImage(img_path.toStdString(), this->img); // CV_8UC3


    // 读取标签文件
    QString gt_path = ui->le_gtpath->text();
    std::vector<std::vector<cv::Point>> contours;
    LoadBoxes(gt_path, contours);


    // 根据4个点，找最小旋转矩形
    cv::RotatedRect box = cv::minAreaRect(contours[0]);
    cv::Point2f center = box.center;
    // 单位是角度
    float angle = box.angle;
    // 前景区域
    int x = (int)(center.x);
    int y = (int)(center.y);
    int w = (int)(box.size.width);
    int h = (int)(box.size.height);
    int x1 = x - w/2;
    int y1 = y - h/2;
    // 左上角点坐标，w,h
    cv::Rect rect_roi(x1,y1,w,h);
    this->rect_roi = rect_roi;
    this->rrect_roi = box;

    // 背景区域
    int padding = (ui->le_bg_ratio->text().toFloat()) * w / 2;
    int x2 = x - w/2 - padding;
    int y2 = y - h/2 - padding;
    int bg_w = w+2*padding;
    int bg_h = h+2*padding;
    // 左上角点坐标，w,h
    cv::Rect rect_bg(x2, y2, bg_w, bg_h);
    this->rect_bg = rect_bg;
    this->rrect_bg = cv::RotatedRect(center, cv::Size2f(box.size.width+2*padding,box.size.height+2*padding), angle);


    // 把两个框都画出来
    int contoursIds = -1;
    cv::Scalar color = cv::Scalar(0,0,255);
    int thickness = 3;
    this->img_result = img.clone();
    // 画前景区域的框
    drawContours(this->img_result, contours, contoursIds, color, thickness);

    std::vector<std::vector<cv::Point>> bg_contours;
    cv::RotatedRect rRect = cv::RotatedRect(cv::Point2f(x,y), cv::Size2f(bg_w, bg_h), angle);
    cv::Point2f vertices[4];      //定义4个点的数组
    rRect.points(vertices);   //将四个点存储到vertices数组中
    // drawContours函数需要的点是Point类型而不是Point2f类型
    std::vector<cv::Point> contour;
    for (int i=0; i<4; i++)
    {
      contour.emplace_back(cv::Point(vertices[i]));
    }
    bg_contours.push_back(contour);

    // 画背景区域的框
    color = cv::Scalar(255,0,0);
    drawContours(this->img_result, bg_contours, contoursIds, color, thickness);


    //获取图像名称和路径
    QFileInfo imginfo = QFileInfo(img_path);
    // 图像名称
    QString img_name = imginfo.fileName();
    //文件后缀
    QString fileSuffix = imginfo.suffix();
    QString save_name = img_name;
    save_name.replace(QRegExp("."+fileSuffix), QString("_roi_bg."+fileSuffix));
    // 保存图像
    QString save_path = ui->le_savepath->text() +"/"+ save_name;
    imwrite(save_path.toStdString(), this->img_result);
    //显示图像
    QImage* srcimg = new QImage;
    srcimg->load(save_path);
    // 图像缩放到label的大小，并保持长宽比
    QImage dest = srcimg->scaled(ui->labelImage->size(),Qt::KeepAspectRatio);
    ui->labelImage->setPixmap(QPixmap::fromImage(dest));


    // 旋转图像
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(img, this->img_rotate, M, cv::Size(img.cols, img.rows), cv::INTER_LINEAR, 0);



    cv::Mat img = this->img.clone();
    // 选择特征
    if (this->fe_status.deep) extract_fe_deep(img, contours);
    if (this->fe_status.gray) extract_fe_gray(img, contours);
    if (this->fe_status.text) extract_fe_texture(img, contours);

}


void Probability::extract_fe_deep(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{
    std::string model_path = "/home/ckq/MyDocuments/QtCode/ftrp_software/sources/model/exp361_no_align_extract_fe_script.pt";

    torch::Device device(torch::kCPU);

    at::Tensor input_tensor;
    cv::Mat img2 = img.clone();
    preprocess(img2, input_tensor);

    input_tensor = input_tensor.to(device);
//     cout << "input img size:" << input_tensor.sizes() << endl;


    torch::NoGradGuard no_grad;
    torch::jit::script::Module model;
    model = torch::jit::load(model_path, device);

//    cout << "导入模型完成"<<endl;
    ui->text_log->append("开始推理...");
    c10::intrusive_ptr<at::ivalue::Tuple> output;
    output = model.forward({input_tensor}).toTuple();

    // 先使用第一个特征图，以后再根据边界框的不同使用不同层级的特征图
    at::Tensor feat = output->elements()[0].toTensor();

    float roi_x = this->rrect_roi.center.x;
    float roi_y = this->rrect_roi.center.y;
    float roi_w = this->rrect_roi.size.width;
    float roi_h = this->rrect_roi.size.height;
    // 弧度
    float roi_theta = (this->rrect_roi.angle / 180) * M_PI;
    // 只有一张图像
    float batch_id =0.0;
    at::Tensor rois = torch::tensor(
                {
                    {0.0, 427.9353,616.8455, 119.1755,14.5517, 1.5757},
//                    {batch_id, roi_x, roi_y, roi_w, roi_h, roi_theta},
                }).to(device);
//    at::Tensor rois = torch::tensor(
//                {
//                    {0.0, 427.9353,616.8455, 119.1755,14.5517, -0.3343},
//                    {0.0, 60.4593, 156.7023, 186.1304, 22.0563, 1.5757}
//                }
//                ).to(device);

    int out_h=7, out_w=7, sample_num=2;
    float spatial_scale = 1/8;
    int chn = feat.sizes()[1];
    int num_rois = rois.sizes()[0];
    at::Tensor outs_rois = torch::zeros({num_rois, chn, out_h, out_w}).to(device);
    roi_align_rotated_forward_cpu(feat, rois, out_h, out_w, spatial_scale, sample_num, outs_rois);
    this->roi_fe.deep = outs_rois;


    float bg_x = this->rrect_bg.center.x;
    float bg_y = this->rrect_bg.center.y;
    float bg_w = this->rrect_bg.size.width;
    float bg_h = this->rrect_bg.size.height;
    // 弧度
    float bg_theta = (this->rrect_bg.angle / 180) * M_PI;
    at::Tensor bgs = torch::tensor(
                {
                    {0.0, 60.4593, 156.7023, 186.1304, 22.0563, 1.5757}
//                    {batch_id, bg_x-400, bg_y+300, bg_w, bg_h, bg_theta},
                }).to(device);


    int num_bgs = bgs.sizes()[0];
    at::Tensor outs_bgs = torch::zeros({num_bgs, chn, out_h, out_w}).to(device);
    roi_align_rotated_forward_cpu(feat, bgs, out_h, out_w, spatial_scale, sample_num, outs_bgs);
    this->bg_fe.deep = outs_bgs;

    ui->text_log->append("获得深度学习特征！");


}

void Probability::extract_fe_gray(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{


    cv::Mat roi = this->img_rotate(this->rect_roi);

//    QString crop_path = save_path.replace(QRegExp("."+fileSuffix), "_crop."+fileSuffix);
//    cv::imwrite((crop_path).toStdString(), roi);

    // 转化为灰度图
//    cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);
//    std::vector<Mat>bgr_plane;
//    cv::split(roi, bgr_plane);
    cv::Mat roi_hist;
    get_hist(roi, roi_hist);


    cv::Mat bg = this->img_rotate(this->rect_bg);
    cv::Mat bg_hist;
    get_hist(bg, bg_hist);

    bg_hist = bg_hist - roi_hist;

//    double roimaxVal=0, bgmaxVal=0;
//    minMaxLoc(roi_hist, 0, &roimaxVal, 0, 0);
//    minMaxLoc(bg_hist, 0, &bgmaxVal, 0, 0);

//    roi_hist = roi_hist / roimaxVal;
//    bg_hist = bg_hist / bgmaxVal;
    at::Tensor roi_hist_tensor = torch::from_blob(roi_hist.data, {roi_hist.rows, roi_hist.cols, roi_hist.channels()});
    at::Tensor bg_hist_tensor = torch::from_blob(bg_hist.data, {bg_hist.rows, bg_hist.cols, bg_hist.channels()});

    float max_roi = roi_hist_tensor.max().item().toFloat();
    float min_roi = roi_hist_tensor.min().item().toFloat();
    float max_bg = bg_hist_tensor.max().item().toFloat();
    float min_bg = bg_hist_tensor.min().item().toFloat();



    roi_hist_tensor = (roi_hist_tensor-max_roi) / (max_roi-min_roi);
    bg_hist_tensor = (bg_hist_tensor-max_bg) / (max_bg-min_bg);


    this->roi_fe.gray = roi_hist_tensor;
    this->bg_fe.gray = bg_hist_tensor;

    ui->text_log->append("获得灰度特征！");

}

void Probability::extract_fe_texture(cv::Mat &img, std::vector<std::vector<cv::Point>> &contours)
{

    cv::Mat img_rotate2 = this->img_rotate.clone();
    cv::cvtColor(img_rotate2, img_rotate2, cv::COLOR_BGR2GRAY);

    // Gabor滤波器参数初始化
    int kernel_size = 3;
    double sigma = 1.0, lambd = CV_PI/8, gamma = 0.5, psi = 0;

    // theta 法线方向
    double theta[8];
    int num_theta = 8;
    for (int i=0; i<num_theta; i++)
    {
        theta[i] = (CV_PI/num_theta) * i;
    }
    // gabor 纹理检测器，可以更多，
    std::vector<cv::Mat> imgs_filtered;
    for(int i = 0; i<num_theta; i++)
    {
        cv::Mat kernel1;
        cv::Mat dest;
        kernel1 = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sigma, theta[i], lambd, gamma, psi, CV_32F);
        filter2D(img_rotate2, dest, CV_32F, kernel1);
        imgs_filtered.push_back(dest);

    }
    // 合并为一个多通道图像
    cv::Mat mc_img;
    cv::merge(imgs_filtered, mc_img);


    cv::Mat roi_fe_text = mc_img(this->rect_roi);
    at::Tensor roi_tensor = torch::from_blob(
                roi_fe_text.data, {roi_fe_text.rows, roi_fe_text.cols, roi_fe_text.channels()});
    at::Tensor roi_tensor_mean = roi_tensor.mean(0).mean(0);


    cv::Mat bg_fe_text = mc_img(this->rect_bg);
    at::Tensor bg_tensor = torch::from_blob(
                bg_fe_text.data, {bg_fe_text.rows, bg_fe_text.cols, bg_fe_text.channels()});
    at::Tensor bg_tensor_mean = bg_tensor.mean(0).mean(0);
    bg_tensor_mean = bg_tensor_mean - roi_tensor_mean;

    // 归一化
    roi_tensor_mean /= roi_tensor_mean.norm(2);
    bg_tensor_mean /= bg_tensor_mean.norm(2);

    this->roi_fe.text = roi_tensor_mean;
    this->bg_fe.text = bg_tensor_mean;


    ui->text_log->append("获得纹理特征！");
}


void Probability::get_hist(cv::Mat & img, cv::Mat & hist)
{
    //    //定义参数变量
        int bins = 8;
        int histSize[] = {bins, bins};
        float bin_range[] = { 0, 255 };
        const float* ranges[] = { bin_range, bin_range };
        int channels[] = {0, 1};

        // 计算得到直方图数据
        calcHist( &img, 1, channels, cv::Mat(), // do not use mask
                 hist, 2, histSize, ranges,
                 true, // the histogram is uniform
                 false );
        /*
        参数解析：
        all_channel[i]:传入要计算直方图的通道，根据函数原函数可以得出，要以引用的方式传入
        1：传入图像的个数，就一个
        0：表示传入一个通道
        Mat():没有定义掩膜，所以默认计算区域是全图像
        b/g/r_hist:用来存储计算得到的直方图数据
        1：对于当前通道需要统计的直方图个数，我们统计一个
        bin:直方图的横坐标有多少个，我们将其赋值为256，即统计每一个像素值的数量。要求用引用方式传入。
        ranges：每个像素点的灰度等级，要求以引用方式传入。
        false：进行归一化，
        false：计算多个图像的直方图时，不累加上一张图像的像素点数据。
        */
}


void Probability::cal_similarity()
{
    // 首先把所有特征相似度归0
    this->fe_similarity = {0.0, 0.0, 0.0};

    if (this->fe_status.deep) cal_similarity_deep();
    if (this->fe_status.gray) cal_similarity_gray();
    if (this->fe_status.text) cal_similarity_text();

    // 根据特征相似度计算综合相似度
    // 使用softmax计算
    float exp_deep = exp(this->fe_similarity.deep);
    float exp_gray = exp(this->fe_similarity.gray);
    float exp_text = exp(this->fe_similarity.text);
    float exp_sum = exp_deep + exp_gray + exp_text;

    float si = 0;
    if (this->fe_status.deep) si = si + exp_deep/exp_sum;
    if (this->fe_status.gray) si = si + exp_gray/exp_sum;
    if (this->fe_status.text) si = si + exp_text/exp_sum;

    this->similarity = si;


}


void Probability::cal_similarity_deep()
{
    cout << "rois:" <<this->rrect_roi.size << endl;
    cout << "bgs:" <<this->rrect_bg.size << endl;
//    cout << "roi_fe:" <<this->roi_fe.deep << endl;
    at::Tensor diff = torch::abs(this->roi_fe.deep - this->bg_fe.deep);
    cout << "sum diff:" << diff.sum() << endl;
    float similarity= 1 - diff.mean().item().toFloat();
    this->fe_similarity.gray = similarity;

    QString log = QString("\n深度学习特征相似度为:") + QString::number(similarity, 'f', 3);
    ui->text_log->append(log);
}


void Probability::cal_similarity_gray()
{
    at::Tensor diff = torch::abs(this->roi_fe.gray - this->bg_fe.gray);
//    cout << "diff_value:" << diff << endl;
    float similarity= 1 - diff.mean().item().toFloat();
    this->fe_similarity.gray = similarity;

    QString log = QString("\n灰度特征相似度为:") + QString::number(similarity, 'f', 3);
    ui->text_log->append(log);
}


void Probability::cal_similarity_text()
{
    at::Tensor diff = torch::abs(this->roi_fe.text - this->bg_fe.text);

    float similarity= 1 - diff.mean().item().toFloat();
    this->fe_similarity.text = similarity;
    QString log = QString("\n纹理特征相似度为:") + QString::number(similarity, 'f', 3);
    ui->text_log->append(log);

}

void Probability::cal_probability()
{

    this->probability = 1 - this->similarity;
    ui->le_probability->setText(QString::number(this->probability));

    QString log = QString("\n计算得到的识别概率为:") + QString::number(this->probability, 'f', 3);
    ui->text_log->append(log);
}

void Probability::save_results()
{
    QString path = ui->le_savepath->text();

    QString img_path = ui->le_imgpath->text();
    QFileInfo imginfo = QFileInfo(img_path);
    QString img_name = imginfo.fileName();
    QString fileSuffix = imginfo.suffix();

    // 保存检测结果
    QString txt_name = img_name;
    txt_name.replace(QRegExp("."+fileSuffix), QString(".txt"));
    QString txt_save_path = path +"/"+ txt_name;

    std::ofstream fout;
    fout.open(txt_save_path.toStdString());

    std::string s = "特征相似度:" + std::to_string(this->similarity) + "\n";
    fout << s;

    s = "目标识别概率:" + std::to_string(this->probability) + "\n";
    fout << s;

    fout.close();

    ui->text_log->append("已完成保存！！！");


}


void Probability::on_bu_fe_group(QAbstractButton *button)
{

    bool status = button->isChecked() ? true : false;
    if (button->text() == QString("深度学习特征")) this->fe_status.deep = status;
    if (button->text() == QString("灰度特征")) this->fe_status.gray = status;
    if (button->text() == QString("纹理特征")) this->fe_status.text = status;

        // 当前点击的按钮
    //    qDebug() << QString("Clicked Button : %1").arg(button->text());

        // 遍历按钮，获取选中状态
//    QList<QAbstractButton*> list = bu_fe_group.buttons();
//    foreach (QAbstractButton *pCheckBox, list)
//    {
//       QString strStatus = pCheckBox->isChecked() ? "Checked" : "Unchecked";
//       qDebug() << QString("Button : %1 is %2").arg(pCheckBox->text()).arg(strStatus);
//    }


}
