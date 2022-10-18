#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numeric>
#include <sys/time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rknn_api.h"
#include "api.h"

void dump_tensor_attr(rknn_tensor_attr *attr)
{
	printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
		   "zp=%d, scale=%f\n",
		   attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
		   attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
		   get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
	unsigned char *data;
	int ret;

	data = NULL;

	if (NULL == fp)
	{
		return NULL;
	}

	ret = fseek(fp, ofst, SEEK_SET);
	if (ret != 0)
	{
		printf("blob seek failure.\n");
		return NULL;
	}

	data = (unsigned char *)malloc(sz);
	if (data == NULL)
	{
		printf("buffer malloc failure.\n");
		return NULL;
	}
	ret = fread(data, 1, sz, fp);
	return data;
}

unsigned char *load_model(const char *filename, int *model_size)
{
	FILE *fp;
	unsigned char *data;

	fp = fopen(filename, "rb");
	if (NULL == fp)
	{
		printf("Open file %s failed.\n", filename);
		return NULL;
	}

	fseek(fp, 0, SEEK_END);
	int size = ftell(fp);

	data = load_data(fp, 0, size);

	fclose(fp);

	*model_size = size;
	return data;
}

rknn_context ctx1;
unsigned char *model_data1 = nullptr;
rknn_input_output_num io_num1;
int left_detect_cnt;  //处理左侧连续帧检测结果
int right_detect_cnt; //处理右侧连续帧检测结果
uint8_t *d2 = new uint8_t[416*416*3];

void reset_containers()
{
	left_detect_cnt = 0;
	right_detect_cnt = 0;
}

// 初始化环境，加载AI模型
int InitEnv(const char *model_path1)
{
	// load yolo model
	printf("Loading yolo mode...\n");
	int model_data_size1 = 0;
	unsigned char *model_data1 = load_model(model_path1, &model_data_size1);
	int ret = rknn_init(&ctx1, model_data1, model_data_size1, 0, NULL);
	if (ret < 0)
	{
		printf("yolo rknn_init error ret=%d\n", ret);
		return -1;
	}
	printf("yolo rknn_init success!\n", ret);
	rknn_sdk_version version;
	ret = rknn_query(ctx1, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
	if (ret < 0)
	{
		printf("rknn_init error ret=%d\n", ret);
		return -1;
	}
	printf(" enviroment init success, sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
	return 0;
}

// yolov5检测嘴部坐标
int YoloDetection(const cv::Mat &src, DetectRes &detect_res)
{
	cv::Mat img;
	cv::cvtColor(src, img, cv::COLOR_BGR2RGB);
	int img_width1 = img.cols;
	int img_height1 = img.rows;
	int ret = rknn_query(ctx1, RKNN_QUERY_IN_OUT_NUM, &io_num1, sizeof(io_num1));
	struct timeval start_time, stop_time;
	if (ret < 0)
	{
		printf("yolo rknn runtime error ret=%d\n", ret);
		return -1;
	}
	// printf("yolo model input num: %d, output num: %d\n", io_num1.n_input, io_num1.n_output);

	rknn_tensor_attr input_attrs1[io_num1.n_input];
	memset(input_attrs1, 0, sizeof(input_attrs1));
	for (int i = 0; i < io_num1.n_input; i++) {
	  input_attrs1[i].index = i;
	  ret                  = rknn_query(ctx1, RKNN_QUERY_INPUT_ATTR, &(input_attrs1[i]), sizeof(rknn_tensor_attr));
	  if (ret < 0) {
	    printf("yolo rknn runtime error ret=%d\n", ret);
	    return -1;
	  }
	  //dump_tensor_attr(&(input_attrs1[i]));
	}

	rknn_tensor_attr output_attrs1[io_num1.n_output];
	memset(output_attrs1, 0, sizeof(output_attrs1));
	for (int i = 0; i < io_num1.n_output; i++) {
	  output_attrs1[i].index = i;
	  ret                   = rknn_query(ctx1, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs1[i]), sizeof(rknn_tensor_attr));
	  //dump_tensor_attr(&(output_attrs1[i]));
	}

	int channel1 = 3;
	int width1 = 0;
	int height1 = 0;
	if (input_attrs1[0].fmt == RKNN_TENSOR_NCHW)
	{
		// printf("yolo model is NCHW input fmt\n");
		channel1 = input_attrs1[0].dims[1];
		width1 = input_attrs1[0].dims[2];
		height1 = input_attrs1[0].dims[3];
	}
	else
	{
		// printf("yolo model is NHWC input fmt\n");
		width1 = input_attrs1[0].dims[1];
		height1 = input_attrs1[0].dims[2];
		channel1 = input_attrs1[0].dims[3];
	}
	// printf("yolo model input height=%d, width=%d, channel=%d\n", height1, width1, channel1);

	rknn_input inputs1[1];
	memset(inputs1, 0, sizeof(inputs1));
	inputs1[0].index = 0;
	inputs1[0].type = RKNN_TENSOR_UINT8;
	inputs1[0].size = width1 * height1 * channel1;
	inputs1[0].fmt = RKNN_TENSOR_NHWC;
	inputs1[0].pass_through = 0;

	// yolov5 input 416*416
	cv::Mat resized;
	// 104 * 104
	cv::resize(img, resized, cv::Size(104, 104));
	cv::Mat bg(cv::Size(width1, height1), CV_8UC3, cv::Scalar(0, 0, 0));
	auto croped = bg(cv::Rect(0, 0, 104, 104));
	cv::addWeighted(croped, 0, resized, 1, 0, croped);
	//int all = bg.channels() * bg.cols * bg.rows;
	for (int i = 0; i < bg.rows; ++i)
	{
		for (int j = 0; j < bg.cols; ++j)
		{
			cv::Vec3b vc = bg.at<cv::Vec3b>(i, j);
			d2[i * bg.cols * 3 + j * 3 + 0] = (uint8_t)vc.val[0];
			d2[i * bg.cols * 3 + j * 3 + 1] = (uint8_t)vc.val[1];
			d2[i * bg.cols * 3 + j * 3 + 2] = (uint8_t)vc.val[2];
		}
	}
	inputs1[0].buf = d2;
	gettimeofday(&start_time, NULL);
	rknn_inputs_set(ctx1, io_num1.n_input, inputs1);

	rknn_output outputs1[io_num1.n_output];
	memset(outputs1, 0, sizeof(outputs1));
	for (int i = 0; i < io_num1.n_output; i++)
	{
		outputs1[i].want_float = 0;
	}

	ret = rknn_run(ctx1, NULL);
	ret = rknn_outputs_get(ctx1, io_num1.n_output, outputs1, NULL);

	// yolo post process
	float scale_w = (float)width1 / img_width1;
	float scale_h = (float)height1 / img_height1;

	detect_result_group_t detect_result_group;
	std::vector<float> out_scales1;
	std::vector<int32_t> out_zps1;
	for (int i = 0; i < io_num1.n_output; ++i)
	{
		out_scales1.push_back(output_attrs1[i].scale);
		out_zps1.push_back(output_attrs1[i].zp);
	}

	post_process((int8_t *)outputs1[0].buf, (int8_t *)outputs1[1].buf, (int8_t *)outputs1[2].buf, height1, width1,
				 BOX_THRESH, NMS_THRESH, scale_w, scale_h, out_zps1, out_scales1, &detect_result_group);
	gettimeofday(&stop_time, NULL);
	printf("yolo once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
	if (detect_result_group.count == 0)
	{
		return 1;
	}

	// top1 mouth only
	detect_result_t *det_result = &(detect_result_group.results[0]);
	printf("%s @ (%d %d %d %d) %f\n", "mouth", det_result->box.left, det_result->box.top,
		   det_result->box.right, det_result->box.bottom, det_result->prop);
	detect_res.mouth_x = det_result->box.left;
	detect_res.mouth_y = det_result->box.top;
	detect_res.mouth_width = det_result->box.right - det_result->box.left;
	detect_res.mouth_height = det_result->box.bottom - det_result->box.top;
	detect_res.mouth_detect_conf = det_result->prop;

	// release memory
	//delete []d2;
	ret = rknn_outputs_release(ctx1, io_num1.n_output, outputs1);
	return ret;
}

//核酸检测流程判断
int CovidDetection(const cv::Mat &src, const DetectParam &param, DetectRes &detect_res, bool left)
{
	cv::Rect area1(param.detect_area_x, param.detect_area_y, param.detect_area_width, param.detect_area_height);
	cv::Rect area2(detect_res.mouth_x, detect_res.mouth_y, detect_res.mouth_width, detect_res.mouth_height);
	//嘴部检测框异常
	if (area1.area() == 0 || area2.area() == 0)
	{
		return 2;
	}
	cv::Rect iou = area1 & area2;
	float iou_f = float(iou.area()) / float(area1.area()); //改为比上检测框的面积，为限制嘴部位置
	//判断嘴部是否在检测区域内
	if (iou_f < param.iou_th)
	{
		reset_containers();
		return 2;
	}

	//检测嘴部张开程度是否合规
	float open_ratio = float(detect_res.mouth_height) / float(detect_res.mouth_width);
	detect_res.open_ratio = open_ratio;
	if (open_ratio < param.open_ratio)
	{
		return 3;
	}

	//检测棉签是否到位
	int unit_len = detect_res.mouth_width / 12.0;
	int left_ctr_x = detect_res.mouth_x;
	int left_ctr_y = detect_res.mouth_y + detect_res.mouth_height / 2.0;
	int right_ctr_x = detect_res.mouth_x + detect_res.mouth_width;
	int right_ctr_y = detect_res.mouth_y + detect_res.mouth_height / 2.0;;

	//显示腺体框
	cv::Rect left_roi(left_ctr_x + unit_len * 1.5, left_ctr_y - unit_len * 0.5, unit_len, unit_len);
	cv::Rect right_roi(right_ctr_x - unit_len * 2.5, right_ctr_y - unit_len * 0.5, unit_len, unit_len);
	left_roi &= cv::Rect(0, 0, src.cols, src.rows);
	if (left_roi.width == 0)
		return 4;
	right_roi &= cv::Rect(0, 0, src.cols, src.rows);
	if (right_roi.width == 0)
		return 4;
	detect_res.l_x = left_roi.x;
	detect_res.l_y = left_roi.y;
	detect_res.l_width = left_roi.width;
	detect_res.l_height = left_roi.height;
	detect_res.r_x = right_roi.x;
	detect_res.r_y = right_roi.y;
	detect_res.r_width = right_roi.width;
	detect_res.r_height = right_roi.height;

	//检测腺体框
	float inner_ratio = param.inner_ratio;
	if (left)
	{
		cv::Rect left_inner_roi(left_roi.x + left_roi.width * inner_ratio, left_roi.y + left_roi.height * inner_ratio * 2,
								left_roi.width * inner_ratio * 2, left_roi.height * inner_ratio * 2);
		left_inner_roi &= cv::Rect(0, 0, src.cols, src.rows);
		if (left_inner_roi.width == 0)
			return 4;
		cv::Mat left_region = src(left_inner_roi).clone();
		cv::Mat left_region_gary;
		cv::cvtColor(left_region, left_region_gary, cv::COLOR_BGR2GRAY);
		cv::Scalar l_mean = cv::mean(left_region_gary);
		float left_mean = l_mean[0];
		detect_res.l_gray_val = left_mean;
		if (left_mean >= param.conf_th)
		{
			if (left_detect_cnt < param.stick_stay_th)
			{
				left_detect_cnt++;
			}
			else
			{
				reset_containers();
				return 5;
			}
		}
	}
	else
	{
		cv::Rect right_inner_roi(right_roi.x + right_roi.width * inner_ratio, right_roi.y + right_roi.height * inner_ratio * 2,
								 right_roi.width * inner_ratio * 2, right_roi.height * inner_ratio * 2);
		right_inner_roi &= cv::Rect(0, 0, src.cols, src.rows);
		if (right_inner_roi.width == 0)
			return 4;
		cv::Mat right_region = src(right_inner_roi).clone();
		cv::Mat right_region_gary;
		cv::cvtColor(right_region, right_region_gary, cv::COLOR_BGR2GRAY);
		cv::Scalar r_mean = cv::mean(right_region_gary);
		float right_mean = r_mean[0];
		detect_res.r_gray_val = right_mean;
		if (right_mean >= param.conf_th)
		{
			if (right_detect_cnt < param.stick_stay_th)
			{
				right_detect_cnt++;
			}
			else
			{
				reset_containers();
				return 6;
			}
		}
	}
	return 0;
}

//循环调用该函数，对每帧数据执行检测
int DoDetection(unsigned char *img, int width, int height, DetectParam param, DetectRes &detect_res, bool left)
{
	cv::Mat src(height, width, CV_8UC3);
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			src.at<cv::Vec3b>(i, j)[0] = img[i * width * 3 + j * 3 + 0];
			src.at<cv::Vec3b>(i, j)[1] = img[i * width * 3 + j * 3 + 1];
			src.at<cv::Vec3b>(i, j)[2] = img[i * width * 3 + j * 3 + 2];
		}
	}
	// cv::Mat src = cv::Mat(height, width, CV_8UC3, (void*)img);
	if (src.empty())
	{
		printf("source img is empty\n");
		return -1;
	}
	int ret = YoloDetection(src, detect_res);
	if (ret != 0)
		return ret;
	ret = CovidDetection(src, param, detect_res, left);
	return ret;
}

//释放内存
int ReleaseEnv()
{
	int ret = 0;

	ret = rknn_destroy(ctx1);

	if (model_data1)
	{
		free(model_data1);
	}
	reset_containers();
	return 0;
}