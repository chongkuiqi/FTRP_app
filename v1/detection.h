#ifndef DETECTION_H
#define DETECTION_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QButtonGroup>

#include "util.h"


namespace Ui {
class Detection;
}

class Detection : public QMainWindow
{
    Q_OBJECT

public:
    explicit Detection(QWidget *parent = nullptr);
    ~Detection();

    void browse_img(QString img_type, bool is_SS);
    void browse_img_opt_SS();
    void browse_img_IR_SS();
    void browse_img_SAR_SS();
    void browse_img_opt_MS();
    void browse_img_IR_MS();
    void browse_img_SAR_MS();

    void browse_save();
    void detect();

    void save_results();

    void switch_mode(const QString &text);
    void switch_mode_show(const QString &text);
    void on_bu_group_SS(QAbstractButton *button);

    void on_bu_group_MS(QAbstractButton *button);

    void get_bu_group_status(QButtonGroup *bu_group, bool is_SS);

    void show_img_results(QString img_type);
    void show_img_opt_results();
    void show_img_IR_results();
    void show_img_SAR_results();

    void save_img(QString img_type);
    void save_img_opt();
    void save_img_IR();
    void save_img_SAR();


    void init_ui();
    void reset_imgpath_show();
    void reset_img_show();
    void reset_show();
    void reset_bu_groups();
    void reset_config_params();

    //路径显示与隐藏
    void widget_ss_show(std::string phase);
    void widget_ss_hide(std::string phase);
    void widget_ms_show(std::string phase);
    void widget_ms_hide(std::string phase);


private:
    Ui::Detection *ui;

    // 检测结果
    std::vector<std::vector<cv::Point>> contours;


    // 单频谱选择按钮组
    QButtonGroup bu_group_SS;
    // 多频谱选择按钮组
    QButtonGroup bu_group_MS;

    // 存储各个特征的状态，即是否提取该特征
    struct_MS<bool> img_status = {false,false, false};

    // 存储各个频谱图像的路径
    struct_MS<QString> img_paths = {QString(""), QString(""), QString("")};


    // 存储各个特征的状态，即是否提取该特征
    struct_MS<cv::Mat> img_results;

};

#endif // DETECTION_H
