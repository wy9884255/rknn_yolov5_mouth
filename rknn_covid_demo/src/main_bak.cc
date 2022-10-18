#include "api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
int main(int argc, char** argv)
{
  //初始化环境
  const char* model1       = "./model/RK356X/face.rknn";
  const char* model2       = "./model/RK356X/hrnet.rknn";
  int ret = 0;
  ret = InitEnv(model1, model2);
  if (ret != 0) {
	printf("%d\n", ret);
	return ret;
  }
  
  //设置检测参数
  DetectParam param;
  param.detect_area_x = 222;
  param.detect_area_y = 540;
  param.detect_area_width = 324;
  param.detect_area_height = 340;
  param.iou_th = 0.7;
  param.open_ratio = 0.65;
  param.frame_th = 10;
  param.inner_ratio = 0.25;
  param.stick_stay_th = 3;
  param.conf_th = 80;
  //调用相机加载图像数据流 type=>unsigned char array
  
  //单张图片
  // char* image_name = "./model/face.jpg";
  // cv::Mat mat_img = cv::imread(image_name, 1);
  // const unsigned char* img = mat_img.data;
  
  // //检测
  // DetectRes detect_res;
  // ret = DoDetection(img, 1280, 1710, param, detect_res);
  // if (ret != 0) return ret;
  
  // //结果展示
  // int x1 = detect_res.face_x;
  // int y1 = detect_res.face_y;
  // int x2 = detect_res.face_x + detect_res.face_width;
  // int y2 = detect_res.face_y + detect_res.face_height;
  // rectangle(mat_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
  // for(int i = 0; i < 68; ++i){
	// int pt_x = detect_res.face_pts[i][0] + x1;
	// int pt_y = detect_res.face_pts[i][1] + y1;
	// cv::circle(mat_img, cv::Point(pt_x, pt_y), 5, cv::Scalar(255, 0, 0), 2);
  // }
  // cv::imwrite("dl.jpg", mat_img);
  
  //视频流
  // cv::VideoCapture capture; 
  // cv::Mat frame;
  // frame = capture.open("test3.mp4");//打开视频流
  // if (!capture.isOpened()){
	  // std::cout<<"can not open"<<std::endl;
	  // return -1 ;
  // }
  
  // std::string output = "output.mp4";
  // int fps = capture.get(cv::CAP_PROP_FPS);
  // int type_video = capture.get(cv::CAP_PROP_FOURCC);
  // int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  // int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  // cv::VideoWriter writer;
  // writer.open(output, type_video, fps, cv::Size(width, height), true);  
  // while (capture.read(frame)) {
	  // int img_width = frame.cols;
	  // int img_height = frame.rows;
	  // const unsigned char* img = frame.data;
	  // DetectRes detect_res;
	  // ret = DoDetection(img, img_width, img_height, param, detect_res);
	  // //display result
	  // int x1 = detect_res.face_x;
	  // int y1 = detect_res.face_y;
	  // int x2 = detect_res.face_x + detect_res.face_width;
	  // int y2 = detect_res.face_y + detect_res.face_height;
	  // rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 3);
	  // for(int i = 0; i < 68; ++i){
		// int pt_x = detect_res.face_pts[i][0] + x1;
		// int pt_y = detect_res.face_pts[i][1] + y1;
		// cv::circle(frame, cv::Point(pt_x, pt_y), 5, cv::Scalar(255, 0, 0), 2);
	  // }
	  // //检测区域
	  // x1 = param.detect_area_x;
	  // y1 = param.detect_area_y;
	  // x2 = param.detect_area_x + param.detect_area_width;
	  // y2 = param.detect_area_y + param.detect_area_height;
	  // rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
	  // //嘴巴外接矩形
	  // x1 = detect_res.mouth_x;
	  // y1 = detect_res.mouth_y;
	  // x2 = detect_res.mouth_x + detect_res.mouth_width;
	  // y2 = detect_res.mouth_y + detect_res.mouth_height;
	  // rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
	  // //张和程度
	  // float open_ratio = detect_res.open_ratio;
	  // cv::putText(frame, std::to_string(open_ratio), cv::Point(100, 100), 1, 2, cv::Scalar(0, 255, 0));
	  // //嘴内检测区域
	  // x1 = detect_res.l_x;
	  // y1 = detect_res.l_y;
	  // x2 = detect_res.l_x + detect_res.l_width;
	  // y2 = detect_res.l_y + detect_res.l_height;
	  // rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 2);
	  
	  // x1 = detect_res.r_x;
	  // y1 = detect_res.r_y;
	  // x2 = detect_res.r_x + detect_res.r_width;
	  // y2 = detect_res.r_y + detect_res.r_height;
	  // rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 2);
	  // //检测结果
	  // std::string res_str = "detect res " + std::to_string(ret);
	  // cv::putText(frame, std::to_string(open_ratio), cv::Point(100, 150), 1, 2, cv::Scalar(0, 0, 255));
	  // writer.write(frame);
  
  // //关闭程序时，释放内存
  // capture.release();
  // ret = ReleaseEnv();
  int idx = 0;
  while (true) {
	  if (idx < 200) {
		  idx++;
		  continue;
	  }
	  if (idx > 300){
		  idx++;
		  break;
	  }
	  int img_width = 720;
	  int img_height = 1280;
	  cv::Mat frame = cv::imread("//sdcard//frames//" + std::to_string(idx) + ".jpg", 1);
	  const unsigned char* img = frame.data;
	  DetectRes detect_res;
	  ret = DoDetection(img, img_width, img_height, param, detect_res);
	  //display result
	  int x1 = detect_res.face_x;
	  int y1 = detect_res.face_y;
	  int x2 = detect_res.face_x + detect_res.face_width;
	  int y2 = detect_res.face_y + detect_res.face_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 3);
	  if (ret == 1) continue;
	  for(int i = 0; i < 68; ++i){
		int pt_x = detect_res.face_pts[i][0];
		int pt_y = detect_res.face_pts[i][1];
		cv::circle(frame, cv::Point(pt_x, pt_y), 5, cv::Scalar(255, 0, 0), 2);
	  }
	  //检测区域
	  x1 = param.detect_area_x;
	  y1 = param.detect_area_y;
	  x2 = param.detect_area_x + param.detect_area_width;
	  y2 = param.detect_area_y + param.detect_area_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
	  //嘴巴外接矩形
	  x1 = detect_res.mouth_x;
	  y1 = detect_res.mouth_y;
	  x2 = detect_res.mouth_x + detect_res.mouth_width;
	  y2 = detect_res.mouth_y + detect_res.mouth_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 2);
	  if (ret == 2) continue;
	  
	  //张和程度
	  float open_ratio = detect_res.open_ratio;
	  cv::putText(frame, std::to_string(open_ratio), cv::Point(100, 100), 1, 3, cv::Scalar(0, 255, 0));
	  if (ret == 3) continue;
	  
	  //嘴内检测区域
	  x1 = detect_res.l_x;
	  y1 = detect_res.l_y;
	  x2 = detect_res.l_x + detect_res.l_width;
	  y2 = detect_res.l_y + detect_res.l_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 2);
	  
	  x1 = detect_res.r_x;
	  y1 = detect_res.r_y;
	  x2 = detect_res.r_x + detect_res.r_width;
	  y2 = detect_res.r_y + detect_res.r_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 2);
	  //左侧检测值
	  std::string l_str = "left res: " + std::to_string(detect_res.l_gray_val);
	  cv::putText(frame, l_str, cv::Point(100, 150), 1, 3, cv::Scalar(0, 255, 0));
	  //右侧检测值
	  std::string r_str = "right res: " + std::to_string(detect_res.r_gray_val);
	  cv::putText(frame, r_str, cv::Point(100, 200), 1, 3, cv::Scalar(0, 255, 0));
	  //检测结果
	  std::string res_str = "detect res: " + std::to_string(ret);
	  cv::putText(frame, res_str, cv::Point(100, 250), 1, 3, cv::Scalar(0, 255, 0));;
      cv::imwrite("//sdcard//output//" + std::to_string(idx++) + ".jpg", frame);
  }
  //关闭程序时，释放内存
  ret = ReleaseEnv();
  return 0;
}
