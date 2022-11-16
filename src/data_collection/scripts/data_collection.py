#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image # TODO: Check data type of subscribers/publishers and import here
from sensor_msgs.msg import JointState
# import rosbag
import csv

class DataCollection:
    def __init__(self):
        # subscriber, publisher
        # TODO: subscribe- edd effector position, camera feed
        self.joint_states_sub = rospy.Subscriber('/my_gen3/joint_states', JointState, self.joint_states_cb)
        self.camera_feed_color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.camera_color_feed_cb)
        self.camera_feed_depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.camera_depth_feed_cb)
        # self.file = open("data_collection.txt", "a")
        # self.bag = rosbag.Bag("data_collection.bag", 'a')
        self.joint_state_f = open("/home/vaidehi/Downloads/Kinova_RoboticArm/src/data_collection/joint_state.csv", 'w')
        self.joint_state_writer = csv.writer(self.joint_state_f)
        self.camera_color_f = open("/home/vaidehi/Downloads/Kinova_RoboticArm/src/data_collection/camera_color_feed.csv", 'w')
        self.camera_color_writer = csv.writer(self.camera_color_f)
        self.camera_depth_f = open("/home/vaidehi/Downloads/Kinova_RoboticArm/src/data_collection/camera_depth_feed.csv", 'w')
        self.camera_depth_writer = csv.writer(self.camera_depth_f)
        
        self.joint_state_f_2 = open("/home/vaidehi/Downloads/Kinova_RoboticArm/src/data_collection/joint_state_2.csv", 'w')
        self.joint_state_writer_2 = csv.writer(self.joint_state_f_2)
        self.camera_color_f_2 = open("/home/vaidehi/Downloads/Kinova_RoboticArm/src/data_collection/camera_color_feed_2.csv", 'w')
        self.camera_color_writer_2 = csv.writer(self.camera_color_f_2)
        self.camera_depth_f_2 = open("/home/vaidehi/Downloads/Kinova_RoboticArm/src/data_collection/camera_depth_feed_2.csv", 'w')
        self.camera_depth_writer_2 = csv.writer(self.camera_depth_f_2)
        
        print("DataCollection")

    def __del__(self):
        # self.file.close()
        self.joint_state_f.close()
        self.camera_color_f.close()
        self.camera_depth_f.close()
        self.joint_state_f_2.close()
        self.camera_color_f_2.close()
        self.camera_depth_f_2.close()
        print("~DataCollection")

    def joint_states_cb(self, msg):
        """header: 
            seq: 1774
            stamp: 
                secs: 1668551773
                nsecs: 509788036
            frame_id: ''
          name: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, finger_joint, left_inner_knuckle_joint,
          left_inner_finger_joint, right_outer_knuckle_joint, right_inner_knuckle_joint, right_inner_finger_joint]
          position: [-0.005732187930286514, 0.4887219180940803, -3.132436439445992, -1.864200246198343, 0.0024580001582711858, 0.7828533132387959, 1.5643789075161325, 0.00698692093770088, 0.00698692093770088, -0.00698692093770088, 0.00698692093770088, 0.00698692093770088, -0.00698692093770088]
          velocity: [-0.002289906319662698, 0.17106830242405116, 0.010021302492616237, 0.3409748008225553, 0.0042357197429195486, -0.1534153726908928, -0.00755036961837568, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0]
          effort: [-0.31865596771240234, 17.228782653808594, 0.03316592797636986, -10.732674598693848, -0.0162641778588295, -2.255537509918213, -0.31152278184890747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        self.end_effector_pos_header = msg.header
        self.end_effector_pos_velocity = msg.velocity
        self.end_effector_pos = msg
        self.joint_state_writer.writerow(self.end_effector_pos_header)
        self.joint_state_writer.writerow(self.end_effector_pos_velocity)
    
    def camera_depth_feed_cb(self, msg):
        self.camera_depth_feed = msg
        self.camera_depth_writer.writerow(self.camera_depth_feed)

    def camera_color_feed_cb(self, msg):
        self.camera_color_feed = msg
        self.camera_color_writer.writerow(self.camera_color_feed)

    def record_data(self):
        self.joint_state_writer_2.writerow(self.end_effector_pos)
        self.camera_depth_writer_2.writerow(self.camera_depth_feed)
        self.camera_color_writer_2.writerow(self.camera_color_feed)


if __name__ == '__main__':
    rospy.init_node('test_node', disable_signals=True)
    data_c = DataCollection()
    # rate = rospy.Rate(10)
    timer_for_one_round_mins = 1 # 1 min
    a = input("How many rounds?")

    for i in range(int(a)):
        now = rospy.get_rostime()
        b = input("Start round %f?", i)
        if(b=='Y' or b=='y'):
            while(rospy.Time.now() != now + rospy.Duration(timer_for_one_round_mins*60)):
                data_c.record_data()
                rospy.sleep(0.01)
        print("round finish")
        
        # ans = input("Save this data?")
        # if(ans=="y" or ans=="Y"):
        #     with rosbag.Bag('final_data.bag', 'w') as outbag:
        #         for topic, msg, t in rosbag.Bag('data_collection.bag').read_messages():
        #             outbag.write(topic, msg, t)
        rospy.sleep(0.1)

    rospy.spin()

    # rospy.signal_shutdown # TODO: Check how to shutdown this

    # while not rospy.is_shutdown():
    #     rospy.sleep(0.01)
