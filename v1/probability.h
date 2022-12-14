#ifndef PROBABILITY_H
#define PROBABILITY_H


#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QButtonGroup>

#undef slots
#include "torch/torch.h"
#include <torch/script.h>
#define slots Q_SLOTS

#include "util.h"


// 泛型编程，存储各个特征的状态、特征相似度
template <typename T>
struct struct_fe
{
    T deep;
    T gray;
    T text;
};


// 图像频谱类型，枚举
enum SpectrumType { OPT_SPECTRUM=0, IR_SPECTRUM, SAR_SPECTRUM};
// 单个频谱所具有的数据
struct SingleSpectrum
{
    // 频谱类型
    enum SpectrumType spectrum_type;

    // 该频谱的图像
    cv::Mat img_1;
    cv::Mat img_2;

    // 图像名称
    std::string img_1_path;
    std::string img_2_path;


    // 该频谱图像提取的各种特征
    struct_fe<at::Tensor> roi_fe_1;
    struct_fe<at::Tensor> roi_fe_2;

    //  该频谱图像各个特征的相似度
    struct_fe<float> fe_similarity = {0.0, 0.0, 0.0};

    // 该频谱的综合特征相似度
    float sim;
};


// 待提取特征区域类型，枚举
// 包括3种，“目标-背景”，“目标-目标（单张图像）”，“目标-目标（两张图像）”
enum RoisType {FG_BG=0, FG_FG, FG_FG2};

// 相似度加权方法，包括特征加权和频谱加权，有自定义法和熵权法两种
enum SimType {CUSTOM=0, ENTROPY};


namespace Ui {
class Probability;
}

class Probability : public QMainWindow
{
    Q_OBJECT

public:
    explicit Probability(QWidget *parent = nullptr);
    ~Probability();

    //路径框的显示与隐藏
    void widget_opt_hide();
    void widget_opt_show();
    void widget_IR_hide();
    void widget_IR_show();
    void widget_SAR_hide();
    void widget_SAR_show();

    void browse_img(SpectrumType img_type, bool is_img_1);
    void browse_img_opt();
    void browse_img_IR();
    void browse_img_SAR();
    void browse_img_opt_2();
    void browse_img_IR_2();
    void browse_img_SAR_2();

    void on_bu_group_MS(QAbstractButton *button);

    void reset_imgpath_show();
    void reset_img_show();
    void reset_MS();



    // 特征提取区域选择
    void on_bu_group_rois(QAbstractButton *button);
    // 显示复位，各个按钮、图像显示全部清零
    void reset_rois_show();
    void reset_rois();

    void browse_gt_1();
    void browse_gt_2();
    void browse_gt_3();
    void browse_gt_4();

    void deal_roi_choose_1(const QString &text);
    void deal_roi_choose_2(const QString &text);
    void deal_roi_choose_3(const QString &text);
    void deal_roi_choose_4(const QString &text);
    void deal_roi_choose_5(const QString &text);
    void show_all_objs_img_1(std::vector<std::vector<cv::Point>> contour_1, std::vector<std::vector<cv::Point>> contour_2);
    void show_all_objs_img_2(std::vector<std::vector<cv::Point>> contour);
    void show_rois_img_1(std::vector<std::vector<cv::Point>> contour_1, std::vector<std::vector<cv::Point>> contour_2);
    void show_rois_img_2(std::vector<std::vector<cv::Point>> contour);

    void extract_rois();



    // 特征选择和提取
    void on_bu_group_fe(QAbstractButton *button);
    void reset_fe();
    void extract_fe();
    void extract_fe_SS(SingleSpectrum &specturm_img);




    // 特征相似度计算和加权
    void on_bu_group_fe_weights(QAbstractButton *button);
    void reset_weights_fe();
    void on_bu_group_sp_weights(QAbstractButton *button);
    void reset_weights_sp();
    void reset_sim_weights();

    void cal_similarity();
    void cal_similarity_SS(SingleSpectrum &specturm_img);


    // 识别概率计算
    void cal_probability();


    void save_results();
    void browse_save();

    void init_ui();


private:
    Ui::Probability *ui;



    float similarity;
    float probability;

    SingleSpectrum opt_spectrum;
    SingleSpectrum ir_spectrum;
    SingleSpectrum sar_spectrum;


    // 存储各个频谱的状态
    struct_MS<bool> img_status = {false,false, false};

    // 多频谱选择按钮组
    QButtonGroup bu_group_MS;




    // 待提取特征区域选择按钮组
    QButtonGroup bu_group_rois;
    // 待提取特征区域类型，包括3种，“目标-背景”，“目标-目标”，“目标-目标2”
    enum RoisType rois_type;
    // 两个contours
    std::vector<std::vector<cv::Point>> contours_1;
    std::vector<std::vector<cv::Point>> contours_2;
    std::vector<cv::Point> contour_1;
    std::vector<cv::Point> contour_2;
    RBox rbox_1;
    RBox rbox_2;




    // 特征选择按钮组
    QButtonGroup bu_group_fe;
    // 存储各个特征的状态，即是否提取该特征
    struct_fe<bool> fe_status = {false,false, false};



    // 相似度加权方式选择按钮组
    // 特征加权
    QButtonGroup bu_group_fe_weights;
    enum SimType weights_type_fe;
    // 存储各个特征的相似度加权权值
    struct_fe<float> fe_sim_weights = {0.0, 0.0, 0.0};
    // 频谱加权
    QButtonGroup bu_group_sp_weights;
    enum SimType weights_type_sp;
    // 存储各个频谱特征的相似度加权权值
    struct_MS<float> spectrum_sim_weights = {0.0, 0.0, 0.0};




    // 映射方法选择按钮组
    QButtonGroup bu_group_map;


};

#endif // PROBABILITY_H
