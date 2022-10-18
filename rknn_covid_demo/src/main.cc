#include "api.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
int main(int argc, char** argv)
{
  //初始化环境
  const char* model1       = "./model/RK3588/mouth.rknn";
  int ret = 0;
  ret = InitEnv(model1);
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
  param.open_ratio = 0.8;
  param.inner_ratio = 0.25;
  param.stick_stay_th = 3;
  param.conf_th = 180;

  bool left_flag = true;
  //调用相机加载图像数据流 type=>unsigned char array
  int idx = 0;
  
  while (true) {
	  if (idx > 294){
		  break;
	  }
	  cv::Mat frame = cv::imread("//sdcard//frames//" + std::to_string(idx) + ".jpg", 1);
	  int img_width = frame.cols;
	  int img_height = frame.rows;
	  //const unsigned char* img = frame.data;
	  unsigned char* img = new unsigned char[img_width*img_height*3];
	  for(int i=0;i<img_height;i++)  
      {  
        for(int j=0;j<img_width;j++)  
        {  
            img[i*img_width*3 + j * 3] = frame.at<cv::Vec3b>(i,j)[0];
            img[i*img_width*3 + j * 3 + 1] = frame.at<cv::Vec3b>(i,j)[1];  
            img[i*img_width*3 + j * 3 + 2] = frame.at<cv::Vec3b>(i,j)[2];  
        }  
      }  
	  DetectRes detect_res; 
	  ret = DoDetection(img, img_width, img_height, param, detect_res, left_flag);
	  delete []img;
	  //嘴部检测框
	  int x1 = detect_res.mouth_x;
	  int y1 = detect_res.mouth_y;
	  int x2 = detect_res.mouth_x + detect_res.mouth_width;
	  int y2 = detect_res.mouth_y + detect_res.mouth_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 3);
	  if (ret == 1) {
		printf("no mouth detect on %d.jpg ", idx++);
	  	continue;
	  }
	  printf("now runing on %d", idx);
	  cv::imwrite("//sdcard//output//" + std::to_string(idx++) + ".jpg", frame);
	  //检测区域
	  x1 = param.detect_area_x;
	  y1 = param.detect_area_y;
	  x2 = param.detect_area_x + param.detect_area_width;
	  y2 = param.detect_area_y + param.detect_area_height;
	  cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
	  continue;
	  if (ret == 2){
		printf("mouth not in right place on %d .jpg", idx++);
		continue;
	  } 
	  
	  //张和程度
	  float open_ratio = detect_res.open_ratio;
	  cv::putText(frame, std::to_string(open_ratio), cv::Point(100, 100), 1, 3, cv::Scalar(0, 255, 0));
	  if (ret == 3) {
		printf("mouth open not ok on %d .jpg", idx++);
		continue;
	  }
	  
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
	  if (ret == 5){
		left_flag = false;
		printf("left part detect finished on %d .jpg", idx++);
		continue;
	  }
	  //右侧检测值
	  std::string r_str = "right res: " + std::to_string(detect_res.r_gray_val);
	  cv::putText(frame, r_str, cv::Point(100, 200), 1, 3, cv::Scalar(0, 255, 0));

	 if (ret == 6){
		printf("检测完毕");
		break;
	  }
	  //检测结果
	  std::string res_str = "detect res: " + std::to_string(ret);
	  cv::putText(frame, res_str, cv::Point(100, 250), 1, 3, cv::Scalar(0, 255, 0));;
      cv::imwrite("//sdcard//output//" + std::to_string(idx++) + ".jpg", frame);
  }
  //关闭程序时，释放内存
  ret = ReleaseEnv();
  return 0;
}
