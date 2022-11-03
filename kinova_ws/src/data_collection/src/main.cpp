// #include <ros/ros.h>
// #include <gtest/gtest.h>
// #include <thread>
// #include <sensor_msgs.h>

// class DataCollection{

// private:
//   ros::NodeHandle nh;
//   ros::Subscriber camera_frames_sub;

//   void camera_cb();

// public:
//   DataCollection()
//   {
//     camera_frames_sub = nh.subscriber<sensor_msgs::Image>("/camera/color/image_rect_color", 1, &DataCollection::camera_cb, this)
//     std::cout << "DataCollection" << std::endl;
//   }

//   ~DataCollection()
//   {
//     std::cout << "~DataCollection" << std::endl;
//   }
// };

// void DataCollection::camera_cb()
// {
//   // TODO: Check data
// }


// int main(int argc, char** argv){
//   ros::init(argc, argv, "KortexArmDriverInitTestsNode");
//   testing::InitGoogleTest(&argc, argv);

//   std::thread t([]{while(ros::ok()) ros::spin();});

//   auto res = RUN_ALL_TESTS();

//   ros::shutdown();
//   return res;
// }